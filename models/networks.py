import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import autocast, float16
import numpy as np


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
        self.input_H, self.input_W = 270, 270
        self.updated_layer_sizes = False
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

        self.features_output_size = [512, int(np.ceil(self.input_H/32)), int(np.ceil(self.input_W/32))]
        self.flattened_size = 16 * 18 * (self.features_output_size[1] - 2) * (self.features_output_size[2] - 2)

        self.potential_fc_layer_sizes = [4096, 2048]
        self.fc_layer_sizes = [self.bound_layer_size(l) for l in self.potential_fc_layer_sizes]
        #self.embedding_to_classification_sizes = [self.embedding_size, 128, 32, 8]

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, self.fc_layer_sizes[0], bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.fc_layer_sizes[0], self.fc_layer_sizes[1], bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.fc_layer_sizes[1], self.embedding_size, bias=False)
        )

        if self.task == "classification":
            if self.num_classes is None:
                raise ValueError("num_classes must be specified for classification task")
            
            # layers = []
            # for layer_size_in, layer_size_out in zip(self.embedding_to_classification_sizes[:-1], self.embedding_to_classification_sizes[1:]):
            #     layers.append(nn.Linear(layer_size_in, layer_size_out, bias=True))
            #     layers.append(nn.ReLU(True))
            #     layers.append(nn.Dropout(0.5))
            # layers.append(
            #     nn.Linear(self.embedding_to_classification_sizes[-1], self.num_classes, bias=True),
            # )
            # self.classifier = nn.Sequential(*layers)
            self.classifier = nn.Linear(self.embedding_size, self.num_classes, bias=True)

    def _calculate_flattened_size(self, input_shape):
        """Calculate flattened_size dynamically based on input dimensions"""
        B, _, _, H, W = input_shape

        self.input_H = H
        self.input_W = W
        
        # Simulate forward pass through features and reducingconvs to get output size
        with torch.no_grad():
            # Create dummy input to calculate dimensions
            dummy_input = torch.zeros(1, 3, H, W)
            
            # Pass through features (ResNet18 backbone)
            dummy_features = self.features(dummy_input)
            _, channels, feat_h, feat_w = dummy_features.shape
            
            # Reshape for 3D convs: [B, 100, channels, feat_h, feat_w]
            dummy_3d = dummy_features.unsqueeze(0).expand(1, 100, channels, feat_h, feat_w)
            
            # Pass through reducing convs
            dummy_reduced = self.reducingconvs(dummy_3d)
            
            # Calculate flattened size
            flattened_size = dummy_reduced.numel() // dummy_reduced.size(0)
            
        return flattened_size
    
    def get_embeddings(self, x, use_autocast=False):
        """Extract embeddings from the model (second-to-last layer)"""
        # x.shape: [B, 300, 1, H, W]
        B, _, _, H, W = x.size()
        if not self.updated_layer_sizes:
            self.update_layer_sizes(H, W)
        
        if use_autocast:
            with autocast('cuda', enabled=True):
                return self._compute_embeddings(x, B, H, W)
        else:
            return self._compute_embeddings(x, B, H, W)
    
    def _compute_embeddings(self, x, B, H, W):
        """Internal method to compute embeddings"""
        # Reshape for ResNet processing
        x = x.view(B * 100, 3, H, W)
        
        # Pass through ResNet features
        x = self.features(x)
        
        # Reshape for 3D conv processing
        channels, feat_h, feat_w = self.features_output_size
        x = x.view(B, 100, channels, feat_h, feat_w)
        
        # Pass through reducing convolutions
        x = self.reducingconvs(x)
        
        # Flatten
        x = x.view(B, self.flattened_size)
        
        # Get embeddings (second-to-last layer)
        x = self.fc(x)
        
        # Normalize embeddings
        return F.normalize(x, p=2, dim=1)
    
    def get_flattened_features(self, x, use_autocast=False):
        """Extract hidden features from the model (flattened 3dConvs)"""
        # x.shape: [self.flattened_size]
        B, _, _, H, W = x.size()
        if not self.updated_layer_sizes:
            self.update_layer_sizes(H, W)
        
        if use_autocast:
            with autocast('cuda', enabled=True):
                return self._compute_flattened_features(x, B, H, W)
        else:
            return self._compute_flattened_features(x, B, H, W)
    
    def _compute_flattened_features(self, x, B, H, W):
        """Internal method to compute flattened 3dConvs"""
        # Reshape for ResNet processing
        x = x.view(B * 100, 3, H, W)
        
        # Pass through ResNet features
        x = self.features(x)
        
        # Reshape for 3D conv processing
        channels, feat_h, feat_w = self.features_output_size
        x = x.view(B, 100, channels, feat_h, feat_w)
        
        # Pass through reducing convolutions
        x = self.reducingconvs(x)
        
        # Flatten
        x = x.view(B, self.flattened_size)
        
        # Return flattened x
        return x

    def forward(self, x):
        # x.shape: [B, 300, 1, H, W]
        if self.task == "embedding":
            return self.get_embeddings(x)
        elif self.task == "classification":
            embeddings = self.get_embeddings(x)
            return self.classifier(embeddings)  # raw logits (no softmax/sigmoid)

    def update_layer_sizes(self, h, w):
        self.input_H, self.input_W = h, w
        self.features_output_size = [512, int(np.ceil(self.input_H/32)), int(np.ceil(self.input_W/32))]
        self.flattened_size = 16 * 18 * (self.features_output_size[1] - 2) * (self.features_output_size[2] - 2)
        self.fc_layer_sizes = [self.bound_layer_size(l) for l in self.potential_fc_layer_sizes]
        self.updated_layer_sizes = True

    def bound_layer_size(self, layer_size):
        if layer_size <= self.flattened_size and layer_size >= self.embedding_size:
            return layer_size
        elif layer_size > self.flattened_size:
            return self.flattened_size
        else:
            return self.embedding_size


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
