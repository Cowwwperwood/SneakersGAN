
class FCGenerator(nn.Module):
    def __init__(self, noise_dim: int, img_shape: tuple):
        super().__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img



class FCDiscriminator(nn.Module):
    def __init__(self, img_shape: tuple):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 256),  
            nn.LeakyReLU(negative_slope=0.01, inplace=True),          
            nn.Linear(256, 256),                                       
            nn.LeakyReLU(negative_slope=0.01, inplace=True),           
            nn.Linear(256, 1),                                         
            nn.Sigmoid(),                                         
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)
