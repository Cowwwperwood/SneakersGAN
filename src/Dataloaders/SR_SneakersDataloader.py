    
class SneakersSRDataset(torch.utils.data.Dataset):  
    def __init__(self, data_dir: str, input_size, target_size):
        super().__init__()
        self.dataset = SneakersDataset(
            data_dir,
            input_size=input_size,
            target_size=target_size,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]  # Returns the (lr, hr) pair

