import torch
import torch.nn as nn

class SegModel(nn.Module):

    def __init__(self, nLayers, inputChannels=3, outputChannels=1, lossWeights=None):
        super(SegModel, self).__init__()
        self.channelNum = 32
        self.resConvBlocksPerLayer = 4

        # First Convolution
        self.conv1 = nn.Conv2d(inputChannels, self.channelNum, (3, 3), padding=1)

        # NN Structure
        self.upsampleBlocks = []
        self.downsampleBlocks = []
        self.preLossConv = []
        for i in range(nLayers):
            self.upsampleBlocks.append(self.createRCBlock(self.resConvBlocksPerLayer))
            self.downsampleBlocks.append(self.createRCBlock(self.resConvBlocksPerLayer))
            self.preLossConv.append(nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(self.channelNum, outputChannels, 3, padding=1)
            ))

        self.upsampleBlocks.append(self.createRCBlock(self.resConvBlocksPerLayer))

        self.upsampleBlocks = nn.ModuleList(self.upsampleBlocks)
        self.downsampleBlocks = nn.ModuleList(self.downsampleBlocks)
        self.preLossConv = nn.ModuleList(self.preLossConv)

        self.losses = [nn.BCEWithLogitsLoss(posWeight=lw, reduction='none') for lw in lossWeights]
        self.losses = nn.ModuleList(self.losses)

    def forward(self, x):
        x = self.conv1(x)

        downSampled = []
        for i, layer in enumerate(self.downsampleBlocks):
            x = layer(x)
            downSampled.append(x)

            x = nn.MaxPool2d(2, stride=2)(x)

        multiscalePred = []
        x = self.upsampleBlocks[0](x)
        x = nn.Upsample(scaleFactor=2.0, mode='nearest')(x)
        for i, layer in enumerate(self.upsampleBlocks[1:]):
            x = x + downSampled[-i - 1]
            x = layer(x)

            multiscalePred.append(self.preLossConv[i](x))
            x = nn.Upsample(scaleFactor=2.0, mode='nearest')(x)
        
        return multiscalePred

    def createRCBlock(self, resConvBlocksPerLayer):
        blocks = []
        for i in range(resConvBlocksPerLayer):
            blocks.append(nn.LeakyReLU(inplace=False))
            blocks.append(nn.Conv2d(self.channelNum, self.channelNum, (3, 3), padding=1))

        return nn.Sequential(*blocks)

    def computeMultiscaleLoss(self, multiscalePred, multiscaleTarget, multiscaleMasks):
        losses = [torch.sum(self.losses[i](x, y) * mask) / torch.sum(mask) for i, (x, y, mask) in
                    enumerate(zip(multiscalePred, multiscaleTarget, multiscaleMasks))]
            
        return sum(losses)

   
        
        