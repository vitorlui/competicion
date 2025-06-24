import torchvision
import torchvision.models as models
import torch
import torch.nn as nn

def get_model_instance(m_name, device, pre_trained=True):

    if pre_trained:

        #["c2plus1","mc3", "r3d", "csn"]
        if(m_name=="c2plus1"):
            model_ = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True).to(device)
            model_.fc = nn.Linear(512, 2, bias=False).to(device)
        elif (m_name=="mc3"):
            model_ = torchvision.models.video.mc3_18(pretrained=True, progress=True).to(device)
            model_.fc = nn.Linear(512, 2, bias=False).to(device)
        elif (m_name=="r3d"):
            model_ = torchvision.models.video.r3d_18(pretrained=True, progress=True).to(device)
            model_.fc = nn.Linear(512, 2, bias=False).to(device)
        elif (m_name=="csn-ir"):
            model_ = ir_csn_152(pretraining="sports1m_ft_kinetics_32frms").to(device)
            model_.fc = nn.Linear(2048, 2, bias=False).to(device)
            #ig_ft_kinetics_32frms
        elif (m_name=="csn-ip"):
            model_ = ip_csn_152(pretraining="sports1m_ft_kinetics_32frms").to(device)
            model_.fc = nn.Linear(2048, 2, bias=False).to(device)

    else:

        if(m_name=="c2plus1"):
            model_ = torchvision.models.video.r2plus1d_18(pretrained=False, progress=True).to(device)
            model_.fc = nn.Linear(512, 2, bias=False).to(device)
        elif (m_name=="mc3"):
            model_ = torchvision.models.video.mc3_18(pretrained=False, progress=True).to(device)
            model_.fc = nn.Linear(512, 2, bias=False).to(device)
        elif (m_name=="r3d"):
            model_ = torchvision.models.video.r3d_18(pretrained=False, progress=True).to(device)
            model_.fc = nn.Linear(512, 2, bias=False).to(device)
        elif (m_name=="csn-ir"):
            model_ = ir_csn_152(pretraining="").to(device)
            model_.fc = nn.Linear(2048, 2, bias=False).to(device)
            #ig_ft_kinetics_32frms
        elif (m_name=="csn-ip"):
            model_ = ip_csn_152(pretraining="").to(device)
            model_.fc = nn.Linear(2048, 2, bias=False).to(device)
        elif (m_name=="csn-ir-18"):
            model_ = ir_csn_18(pretraining="").to(device)
            model_.fc = nn.Linear(2048, 2, bias=False).to(device)
        elif (m_name=="csn-ip-18"):
            model_ = ip_csn_18(pretraining="").to(device)
            model_.fc = nn.Linear(2048, 2, bias=False).to(device)

    return model_

def test():
    print("test OK")
