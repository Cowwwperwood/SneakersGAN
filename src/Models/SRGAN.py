class ResidualBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding='same'), 
            nn.ReLU(),
            nn.BatchNorm2d(filters, momentum=0.8),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(filters, momentum=0.8),
        )

    def forward(self, z):
        return self.model(z) + z


class UpsamplingBlock(nn.Module):
    def __init__(self, filters: int, input_filters: int = None):
        super().__init__()
        input_filters = input_filters or filters  
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),  
            nn.Conv2d(input_filters, filters, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.model(z)


class SRGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=9 // 2, padding_mode="zeros"),
            nn.ReLU(inplace=True),
        )
        self.residual_chain = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)],
            nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2, padding_mode="zeros"),
            nn.BatchNorm2d(64, momentum=0.8)
        )
        self.upsample_conv = nn.Sequential(
            UpsamplingBlock(256, input_filters=64),
            UpsamplingBlock(256),
            nn.Conv2d(256, 3, kernel_size=9, padding=9 // 2, padding_mode="zeros"),
            nn.Tanh(),
        )

    def forward(self, z):
        x = z.permute(0, 3, 1, 2)
        conv = self.init_conv(x)
        x = self.residual_chain(conv) + conv
        img = self.upsample_conv(x)
        img = img.permute(0, 2, 3, 1)
        return img





class DBlock(nn.Module):
    def __init__(
        self,
        input_filters: int,
        filters: int,
        strides: int,
        batch_norm: bool = False,
    ):
        super().__init__()
        padding = (3 - 1) // 2 
        if strides != 1:  
            padding = ((strides - 1) + (3 - 1) // 2)  

        layers = [
            nn.Conv2d(input_filters, filters, kernel_size=3, stride=strides, padding=padding),  
            nn.LeakyReLU(0.2),  
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(filters, momentum=0.8))  
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)
      

class SRDiscriminator(nn.Module):
    def __init__(self, input_size=(112, 112)):
        super().__init__()
        self.input_size = input_size  
        self.model = nn.Sequential(
            DBlock(input_filters=3, filters=64, strides=1, batch_norm=True),
            DBlock(input_filters=64, filters=64, strides=2),
            DBlock(input_filters=64, filters=128, strides=1, batch_norm=True),
            DBlock(input_filters=128, filters=128, strides=2),
            nn.Flatten(),
            nn.LeakyReLU(0.2),
        )
        
        self._calculate_fc_input_size()
        
        self.fc = nn.Linear(self.fc_input_size, 1024)
        self.output_layer = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def _calculate_fc_input_size(self):
        x = torch.randn(1, 3, *self.input_size)
        x = self.model(x) 
        self.fc_input_size = x.size(1)  

    def forward(self, z):
        x = z.permute(0, 3, 1, 2)  
        x = self.model(x) 
        x = self.fc(x)  
        x = self.output_layer(x) 
        return self.sigmoid(x)