
class DCGenerator(nn.Module):
    def __init__(self, noise_dim: int, img_shape: tuple):
        super().__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128 * 7 * 7),
            nn.ReLU(),
            nn.BatchNorm1d(128 * 7 * 7, momentum=0.8),
            
            nn.Unflatten(1, (128, 7, 7)),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128, momentum=0.8),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64, momentum=0.8),
            
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.permute(0, 2, 3, 1)
        return img



class DCDiscriminator(nn.Module):
    def __init__(self, img_shape: tuple):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[2], 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )

    def forward(self, img):
        img_chan = img.permute(0, 3, 1, 2)  
        return self.model(img_chan)


