import torch 
import matplotlib.pyplot as plt


if __name__ == "__main__":
    bigrams = {}
    words = open('names.txt').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    N = torch.zeros((28, 28), dtype=torch.int32)
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1

    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')
    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
            plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')     
    # plt.axis('off')
    # plt.show()
    
    p = N[0].float()
    p /= p.sum()
    # we will have a row and column of zeors since the <E> will never be the start character and the <S> will never be the end character
    g = torch.Generator().manual_seed(2147483647)
    p = torch.rand(3, generator=g)
    p /= p.sum()
    print(p)  # probability for each character to be the first one
    torch.multinomial(p, num_samples=20, replacement=True, generator=g)
    
