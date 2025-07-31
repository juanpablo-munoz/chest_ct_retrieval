import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import autocast, float16


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Proximity100x100(nn.Module):
    def __init__(self, embedding_size: int, num_classes: int = None, task: str = "embedding"):
        """
        task: 'embedding' or 'classification'
        """
        super(Proximity100x100, self).__init__()
        assert task in ["embedding", "classification"]
        self.task = task
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        resnet = models.resnet18(weights='DEFAULT')
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))

        self.reducingconvs = nn.Sequential(
            nn.Conv3d(100, 64, kernel_size=(3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(True),
            nn.Conv3d(64, 32, kernel_size=(3,1,1), stride=(3,1,1), padding=0),
            nn.ReLU(True),
            nn.Conv3d(32, 16, kernel_size=(3,1,1), stride=(3,1,1), padding=0),
            nn.ReLU(True)
        )

        self.flattened_size = 16 * 18 * 3 * 3
        #self.flattened_size = 16 * 18 * 1 * 1

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_size, bias=False)
        )

        if self.task == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be specified for classification task")
            self.classifier = nn.Linear(embedding_size, num_classes, bias=True)

    def forward(self, x):
        # x.shape: [B, 300, 1, H, W]
        #print('(input) x.shape:', x.shape)
        B, _, _, H, W = x.size()
        x = x.view(B * 100, 3, H, W)
        #print('(reshaped) x.shape:', x.shape)
        #self.replace_bn_with_instancenorm(self.features, device=self.features[0].weight.device)
        #self.features.eval()
        #for param in self.features.parameters():
        #    param.requires_grad = False
        x = self.features(x)
        x = x.view(B, 100, 512, 5, 5)
        #print('features(x).shape:', x.shape)
        x = self.reducingconvs(x)
        #print('reducingconvs(x).shape:', x.shape)
        x = x.view(B, self.flattened_size)
        #print('(reshaped) reducingconvs(x).shape:', x.shape)
        x = self.fc(x)
        #print('fc(x).shape:', x.shape)
        if self.task == "embedding":
            return F.normalize(x, p=2, dim=1)
        elif self.task == "classification":
            x = F.normalize(x, p=2, dim=1)
            return self.classifier(x)  # raw logits (no softmax/sigmoid)
    
    def replace_bn_with_instancenorm(self, module, device):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                inst_norm = nn.InstanceNorm2d(child.num_features, affine=True).to(device)
                setattr(module, name, inst_norm)
            else:
                self.replace_bn_with_instancenorm(child, device)



class Proximity300x300(nn.Module):
    def __init__(self, embedding_size: int):
        super(Proximity300x300, self).__init__()
        
        resnet = models.resnet18(weights='DEFAULT')
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))

        #conv input torch.Size([1,83,512,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(100, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16*18*3*3, 512, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256, bias=False), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(256, embedding_size, bias=False)
        )
      
    def forward(self, x):
        # input.shape = [batch_size, 300, 1, h, w]
        shape = list(x.size())
        #print("input.shape:", shape)
        batch_size = int(shape[0])
        h, w = x.size()[-2:]
        x = x.view(batch_size*100, 3, h, w) # x is 5D but Resnet expects a 4D input - so, let's squeeze the first dimension!
        #print("squeezed input.shape:", list(x.size()))
        x = self.features(x)
        #print('resnet output shape:', x.size())
        x = x.view(batch_size,100,512,10,10) # Now, we bring back the first dimension for the 3DConvs!
        x = self.reducingconvs(x)
        #reducingconvs output shape: torch.Size([batch_size, 16, 18, 3, 3])
        #print('reducingconvs output shape:', x.size())
        x = x.view(batch_size, 16*18*3*3) # Flatten all except the first dimension
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class SynthDataEmbeddingModel300x300(nn.Module):
    def __init__(self, embedding_size: int):
        super(SynthDataEmbeddingModel300x300, self).__init__()
        
        resnet = models.resnet18(weights='DEFAULT')
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))

        #conv input torch.Size([1,83,512,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(100, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16*18*3*3, 512, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256, bias=False), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(256, embedding_size, bias=False)
        )
      
    def forward(self, x):
        # input.shape = [batch_size, 300, 1, h, w]
        shape = list(x.size())
        batch_size = int(shape[0])
        h, w = x.size()[-2:]
        x = x.view(batch_size*100, 3, h, w) # x is 5D but Resnet expects a 4D input - so, let's squeeze the first dimension!
        x = self.features(x)
        #print('resnet output shape:', x.size())
        x = x.view(batch_size,100,512,10,10) # Now, we bring back the first dimension for the 3DConvs!
        x = self.reducingconvs(x)
        #reducingconvs output shape: torch.Size([batch_size, 16, 18, 3, 3])
        #print('reducingconvs output shape:', x.size())
        x = x.view(batch_size, 16*18*3*3) # Flatten all except the first dimension
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class SynthDataEmbeddingModel100x100(nn.Module):
    def __init__(self, embedding_size: int):
        super(SynthDataEmbeddingModel100x100, self).__init__()
        
        resnet = models.resnet18(weights='DEFAULT')
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))

        #conv input torch.Size([1,83,512,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(100, 16, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(True),
            
            nn.Conv3d(16, 8, kernel_size = (3,1,1), stride=(3,1,1), padding=0),
            nn.ReLU(True),
            
            nn.Conv3d(8, 4, kernel_size = (3,1,1), stride=(3,1,1), padding=0),
            nn.ReLU(True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(4*18*2*2, 256, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 256, bias=False), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(256, embedding_size, bias=False)
        )
      
    def forward(self, x):
        # input.shape = [batch_size, 300, 1, h, w]
        shape = list(x.size())
        batch_size = int(shape[0])
        h, w = x.size()[-2:]
        x = x.view(batch_size*100, 3, h, w) # x is 5D but Resnet expects a 4D input - so, let's squeeze the first dimension!
        x = self.features(x)
        #print('resnet output shape:', x.size())
        x = x.view(batch_size,100,512,4,4) # Now, we bring back the first dimension for the 3DConvs!
        x = self.reducingconvs(x)
        #output is shape [4, 18, 2, 2]
        #print('reducingconvs output shape:', x.size())
        x = x.view(batch_size, 4*18*2*2) # Flatten all except the first dimension
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
        

class CTPreEmbeddedModel(nn.Module):
    def __init__(self, embedding_size):
        super(CTPreEmbeddedModel, self).__init__()

        '''
        Intended to be used on pre-calculated ResNet18 hidden states.
        Input size = [batch_size, 512, 10, 10]
        Output size = [batch_size, embedding_size]
        '''

        
        #resnet = models.resnet18(weights='DEFAULT')
        #self.features = nn.Sequential(*(list(resnet.children())[:-2]))

        #conv input torch.Size([1,83,512,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(100, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16*18*3*3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, embedding_size))
      
    def forward(self, x):
        shape = list(x.size())
        batch_size = int(shape[0])
        x = x.view(batch_size,100,512,10,10)
        #print('reducingconvs input:', x.size())
        x = self.reducingconvs(x)
        #print('reducingconvs output:', x.size())
        x = x.view(batch_size, 16*18*3*3)
        x = self.fc(x)
        return x


class CTEmbeddingClasificationModel(nn.Module):
    def __init__(self):
        super(CTEmbeddingModel, self).__init__()
        
        #resnet = models.resnet18(weights='DEFAULT')
        #self.features = nn.Sequential(*(list(resnet.children())[:-2]))

        #conv input torch.Size([1,83,512,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(100, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(32, 16, kernel_size = (3,1,1), stride=(3,1,1), padding=0),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16*18*6*6, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(256, embedding_size))
      
    def forward(self, x):
        #print('input shape', x.shape)
        #x = x.permute(0, 3, 1, 2)
        #print(x.shape)
        shape = list(x.size())
        #example shape: [1,83,3,290,290]
        #example shape: [2,83,3,290,290]
        batch_size = int(shape[0])
        x = x.view(batch_size*28,3,130,130)
        x = self.features(x)
        #print('resnet output shape:', x.size())
        x = x.view(batch_size,28,512,5,5)
        x = self.reducingconvs(x)
        #output is shape [batch_size, 16, 18, 3, 3]
        #print('reducingconvs output shape:', x.size())
        x = x.view(batch_size, 16*170*3*3)
        x = self.fc(x)
        return x
        

class CTEmbeddingModel(nn.Module):
    def __init__(self):
        super(CTEmbeddingModel, self).__init__()
        
        resnet = models.resnet18(weights='DEFAULT')
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))

        #conv input torch.Size([1,83,512,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(28, 16, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            #nn.Conv3d(16, 8, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            #nn.ReLU(),
            
            #nn.Conv3d(8, 4, kernel_size = (3,1,1), stride=(3,1,1), padding=0),
            #nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16*170*3*3, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 2))
      
    def forward(self, x):
        #print('input shape', x.shape)
        #x = x.permute(0, 3, 1, 2)
        #print(x.shape)
        shape = list(x.size())
        #example shape: [1,83,3,290,290]
        #example shape: [2,83,3,290,290]
        batch_size = int(shape[0])
        x = x.view(batch_size*28,3,130,130)
        x = self.features(x)
        #print('resnet output shape:', x.size())
        x = x.view(batch_size,28,512,5,5)
        x = self.reducingconvs(x)
        #output is shape [batch_size, 16, 18, 3, 3]
        #print('reducingconvs output shape:', x.size())
        x = x.view(batch_size, 16*170*3*3)
        x = self.fc(x)
        return x
        

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, embedding_size, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear1 = nn.PReLU()
        self.fc1 = nn.Linear(embedding_size, 2)
        self.nonlinear2 = nn.PReLU()
        self.fc2 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear1(output)
        output = self.fc1(output)
        output = self.nonlinear2(output)
        scores = self.fc2(output)
        return scores

    def get_embedding(self, x):
        return self.nonlinear2(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        #print('[x1.shape, x2.shape, x2.shape]:', [x1.size(), x2.size(), x2.size()])
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return torch.cat(output1, output2, output3)

    def get_embedding(self, x):
        return self.embedding_net(x)
