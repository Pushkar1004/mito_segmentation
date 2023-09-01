import torch
import torch.nn as nn
import torch.optim as optim

#ConvolutionBlock

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding= "same")
        self.batchnorm1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding = "same")
        self.batchnorm2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)    
        return x;

    
#EncoderBlock
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.maxpool = nn.MaxPool3d(2)
        
    def forward(self, inputs):
        x = self.conv_block(inputs)
        p = self.maxpool(x)
        return x,p;
    
    
#DecoderBlock
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size= 2 , stride = 2, padding = 0)
        self.conv_block = ConvBlock(out_channels+out_channels, out_channels)   
        
    def forward(self, inputs, skip_connection):
        x = self.conv_transpose(inputs)
        x = torch.cat([x, skip_connection], axis = 1)
        x = self.conv_block(x)
        return x;     
    
    
#Unetwork Architecture 
    
class UNet(nn.Module):
    def __init__(self, num_classes, input_shape = (1,64,704,704 )):
        super(UNet, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.encoder1 = EncoderBlock(1,8)           
        self.encoder2 = EncoderBlock(8,16) 
        self.encoder3 = EncoderBlock(16,32)
        self.encoder4 = EncoderBlock(32,64)
        self.bottleneck = ConvBlock(64,128)
        self.decoder4 = DecoderBlock(128,64)
        self.decoder3 = DecoderBlock(64,32)
        self.decoder2 = DecoderBlock(32,16)
        self.decoder1 = DecoderBlock(16,8)
        self.output_conv = nn.Conv3d(8, num_classes, kernel_size=1 , padding = "same")
        
        if num_classes == 1:
            self.activation = nn.Sigmoid() 
        else:
            self.activation = nn.Softmax(dim=1)   
            


    def forward(self, x):   
        s1, p1 = self.encoder1(x) 
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)
        b1 = self.bottleneck(p4)
        d4 = self.decoder4(b1,s4)
        d3 = self.decoder3(d4,s3)
        d2 = self.decoder2(d3,s2)
        d1 = self.decoder1(d2,s1)
        output = self.output_conv(d1)
        output = self.activation(output)       
        return output