#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, sys, random
import time, itertools, importlib

from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, speaker_label=None,accent_label=None,old_model_label=None,type_output=None):
        return self.module(x, speaker_label,accent_label,old_model_label,type_output)


class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, s_state=None,logger=None, **kwargs):
        super(SpeakerNet, self).__init__()

        self.logger=logger
        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        #TODO make change output size of model 
        self.__S1__ = SpeakerNetModel(**kwargs)
        missing, unexpected = self.__S1__.load_state_dict(s_state, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        self.speaker_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256)  # residual added in forward
        )
        
        self.accent_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256)  # residual added in forward
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512)  # residual added in forward
        )

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        LossFunctionDistillation = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        #TODO make change input size of loss 
        self.__L1__ = LossFunction(**kwargs)
        self.__L2__ = LossFunction(**kwargs)
        #TODO Distillation to teacher (for fusion): cosine similarity + L2 to t
        self.__L3__ = LossFunctionDistillation(**kwargs)
        self.nPerSpeaker = nPerSpeaker
        self.it=0

    def forward(self, data, speaker_label=None,accent_label=None,old_model_label=None,type_output=None):

        data = data.reshape(-1, data.size()[-1]).cuda()
        with torch.no_grad():
            outp = self.__S1__.forward(data)
        

        # assert speaker_label is not None and accent_label is not None, "must have label for both speaker and accent"
        # print("speaker_label:",speaker_label,"accent_label:",accent_label)
        # assert old_model_label is not None , "Must have true H/ASP model"
        
        
        output1 = outp          
        output2 = outp           
         
        speaker_head_out = self.speaker_head(output1) 
        accent_head_out  = self.accent_head(output2) 

        if type_output=='speaker':
            return speaker_head_out
        if type_output=='accent':
            return speaker_head_out

        speaker_head_out = speaker_head_out.reshape(self.nPerSpeaker, -1, speaker_head_out.size()[-1]).transpose(1, 0).squeeze(1)

        nloss_speaker, prec_speaker = self.__L1__.forward(speaker_head_out, speaker_label)

        accent_head_out = accent_head_out.reshape(self.nPerSpeaker, -1, accent_head_out.size()[-1]).transpose(1, 0).squeeze(1)
        
        nloss_accent, prec_accent = self.__L2__.forward(accent_head_out, accent_label)
        
        
        fused_input = torch.cat([speaker_head_out, accent_head_out], dim=-1)
        
        fusion_out = self.fusion(fused_input)
        
        nloss_fusion, prec_fusion = self.__L3__.forward(fusion_out,old_model_label)
        
        self.logger.add_scalar('nloss_speaker', nloss_speaker, self.it)
        self.logger.add_scalar('nloss_accent', nloss_accent, self.it)
        self.logger.add_scalar('nloss_fusion', nloss_fusion, self.it)
        self.it=self.it+1
        
        return nloss_speaker+nloss_accent+nloss_fusion, (prec_speaker+prec_accent+prec_fusion)/3


class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader,oldModel, verbose):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0
        # EER or accuracy

        tstart = time.time()

        for data, data_speaker_label,data_accent_label in loader:
            old_model_label=oldModel(data)
            # print("output_old_model:",output_old_model)
            data = data.transpose(1, 0)

            self.__model__.zero_grad()

            speaker_label = torch.LongTensor(data_speaker_label).cuda()
            accent_label = torch.LongTensor(data_accent_label).cuda()

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss, prec1 = self.__model__(data, speaker_label,accent_label,old_model_label)
                nloss.backward()
                self.__optimizer__.step()

            loss += nloss.detach().cpu().item()
            top1 += prec1.detach().cpu().item()
            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                sys.stdout.write("Loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz ".format(loss / counter, top1 / counter, stepsize / telapsed))
                sys.stdout.flush()

            if self.lr_step == "iteration":
                self.__scheduler__.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        return (loss / counter, top1 / counter)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=100, num_eval=10,type_output=None, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        ## Read all lines
        
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()
        
        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        ## Extract features for every image
        
        
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat = self.__model__(inp1,type_output=type_output).detach().cpu()
            feats[data[1][0]] = ref_feat
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat.size()[1])
                )

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)

        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines):

                data = line.split()

                ## Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                ref_feat = feats[data[1]].cuda()
                com_feat = feats[data[2]].cuda()

                Loss = None
                if type_output=='speaker':
                    Loss=self.__model__.module.__L1__
                if type_output=='accent':
                    Loss=self.__model__.module.__L2__
                if Loss.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                dist = torch.cdist(ref_feat.reshape(num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()

                score = -1 * numpy.mean(dist)

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()

        return (all_scores, all_labels, all_trials)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):

        torch.save(self.__model__.module.state_dict(), path)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)
