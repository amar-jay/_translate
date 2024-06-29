import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
from model import TConfig, Transformer, Encoder, Decoder


class Trainer:
    def __init__(self, model:Transformer, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    def train(self, train_loader, epoch):

        self.model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            logits, loss = self.model(inputs, targets)
            output_dim = logits.shape[-1]

            #flatten and omit SOS from target
            logits = logits.contiguous().view(-1, output_dim)

            loss.backward()

            self.optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('--block_size', type=int, default=124)
        parser.add_argument('--vocab_size', type=int, default=100)
        parser.add_argument('--n_layer', type=int, default=12)
        parser.add_argument('--n_head', type=int, default=12)
        parser.add_argument('--n_embd', type=int, default=12)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--bias', type=bool, default=True)
        parser.add_argument('--padding_idx', type=int, default=0)

        parser.add_argument('--lr', type=float, default=0.0001)

        parsed_args = parser.parse_args()
        block_size = parsed_args.block_size
        vocab_size = parsed_args.vocab_size
        n_layer = parsed_args.n_layer
        n_head = parsed_args.n_head
        n_embd = parsed_args.n_embd
        dropout = parsed_args.dropout
        bias = parsed_args.bias
        padding_idx = parsed_args.padding_idx

        lr = parsed_args.lr

        config = TConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
            padding_idx=padding_idx
        )

        encoder_part = Encoder(config)
        decoder_part = Decoder(config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {device} device')

        model = Transformer(encoder_part, decoder_part, config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(model, optimizer, device)
        print('NOTE: Using completely random data as dataset')
        # dataset shiuld be a torchdata.Dataset with (input, target) pairs of shape (1, block_size) and type torch.LongTensor
        dataset = torchdata.TensorDataset(torch.randint(0, vocab_size, (1, block_size)), torch.randint(0, vocab_size, (1, block_size)))
        train_loader = torchdata.DataLoader(dataset)

        for epoch in range(10):
            trainer.train(train_loader, epoch)

        print('Finished Training')



