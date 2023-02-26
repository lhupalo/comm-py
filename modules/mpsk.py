import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

def qpsk(num_bits, EbN0_dBs):
    """Realiza simulação de uma transmissão QPSK através de um canal AWGN.

    Args:
        num_bits (integer): Número de bits de informação a serem utilizados na simulação.
        EbN0_dBs (array): Array de medidas da relação entre a energia média transmitida por bit e a densidade espectral de potência do ruído aditivo branco gaussiano (AWGN) no canal de comunicação

    Returns:
        BER (array): Taxa de erro de bit simulada
        SER (array): Taxa de erro de símbolo simulada
    """    

    num_symbols = int(num_bits/2)
        
    num_errors_bit = np.zeros(len(EbN0_dBs))
    num_errors_symb = np.zeros(len(EbN0_dBs))

    for i in range(len(EbN0_dBs)):

        # Vetor de bits a ser transmitido
        x = np.round(np.random.rand(num_bits))

        # Modulação do sinal
        j = np.arange(0, len(x), 2)
        mask1 = np.logical_and(x[j] == 0, x[j+1] == 0)
        mask2 = np.logical_and(x[j] == 0, x[j+1] == 1)
        mask3 = np.logical_and(x[j] == 1, x[j+1] == 1)
        mask4 = np.logical_and(x[j] == 1, x[j+1] == 0)

        rl = np.zeros(len(j))
        img = np.zeros(len(j))

        rl[mask1] = 1
        img[mask1] = 0

        rl[mask2] = 0
        img[mask2] = 1

        rl[mask3] = -1
        img[mask3] = 0

        rl[mask4] = 0
        img[mask4] = -1

        # Sinal transmitido obtido através da modulação do vetor de bits de informação
        s = rl + 1j*img
        
        # Variância do sinal transmitido e adição de ruído AWGN
        var = (1/np.sqrt(4))*10**(-EbN0_dBs[i]/20)

        noise_phase = var * np.random.randn(num_symbols)
        noise_quad = var * np.random.randn(num_symbols)

        # Sinal recebido é o sinal transmitido + ruído
        r_phase = np.real(s) + noise_phase
        r_quad = np.imag(s) + noise_quad
        
        r = r_phase + 1j*r_quad

        # Demodulação do sinal recebido
        phi = np.mod(np.arctan2(r_quad, r_phase) * 180/np.pi + 360, 360)


        d = []
        d_s = []
        for j in range(len(phi)):
            
            if (bool(phi[j] >= 0) & bool(phi[j] <= 45)) | bool(phi[j] > 315): # Símbolo 1 -> 00
                b1 = 0
                b2 = 0
                c = complex(b1,b2)
            elif (bool(phi[j] > 45) & bool(phi[j] < 135)): # Símbolo 2 -> 01
                b1 = 0
                b2 = 1
                c = complex(b1,b2)
            elif (bool(phi[j] > 135) & bool(phi[j] < 225)): # Símbolo 3 -> 11
                b1 = 1
                b2 = 1
                c = complex(b1,b2)
            elif (bool(phi[j] > 225) & bool(phi[j] < 315)): # Símbolo 4 -> 10
                b1 = 1
                b2 = 0
                c = complex(b1,b2)
        
            d.append([b1, b2])
            d_s.append(c)

        
        # Detecção de bits
        num_errors_bit[i] = np.sum(np.array(d).reshape(-1) != x)

        # Detecção de símbolos
        num_errors_symb[i] = np.count_nonzero(x.reshape(-1,2) != np.array(d))

    """
    - BER = Bit Error Rate
    - SER = Symbol Error Rate

    Neste caso, para cada dois bits de informação existe o mapeamento em um símbolo para transmissão.
    """

    BER = num_errors_bit/num_bits
    SER = num_errors_symb/num_symbols

    return BER, SER


def bpsk(num_bits, EbN0_dBs):
    """Realiza simulação de uma transmissão BPSK através de um canal AWGN.

    Args:
        num_bits (integer): Número de bits de informação a serem utilizados na simulação.
        EbN0_dBs (array): Array de medidas da relação entre a energia média transmitida por bit e a densidade espectral de potência do ruído aditivo branco gaussiano (AWGN) no canal de comunicação

    Returns:
        BER (array): Taxa de erro de bit simulada
        SER (array): Taxa de erro de símbolo simulada
    """ 

    num_errors = np.zeros(len(EbN0_dBs))

    for i in range(len(EbN0_dBs)):

        # Vetor de bits a ser transmitido
        x = np.round(np.random.rand(num_bits))

        # Sinal transmitido obtido através da modulação do vetor de bits de informação
        s = np.round(2*x-1)

        # Variância do sinal transmitido e adição de ruído AWGN
        var = (1/np.sqrt(2))*10**(-EbN0_dBs[i]/20)
        noise = var * np.random.randn(num_bits)

        # Sinal recebido é o sinal transmitido + ruído
        r = s + noise

        # Demodulação do sinal recebido
        d = np.sign(r).astype(int)
        
        # Detecção do sinal demodulado    
        num_errors[i] = np.count_nonzero(d != s)
    
    # No caso BPSK, pelo fato do número de símbolos ser o mesmo do número de bits, a BER é igual à SER
    BER = num_errors/num_bits
    SER = BER
    
    return BER, SER

def PlotBER_SER(EbN0_dBs, BER, SER, M):
    """Faz o plot das curvas de BER e SER simuladas e teóricas, tendo como entrada os valores da simulação.

    Args:
        EbN0_dBs (array): Array de medidas da relação entre a energia média transmitida por bit e a densidade espectral de potência do ruído aditivo branco gaussiano (AWGN) no canal de comunicação
        BER (array): Taxa de erro de bit simulada
        SER (array): Taxa de erro de símbolo simulada
        M (integer): Ordem da modulação. Ex.: BPSK -> M = 2, 2-PSK
    """    
    
    if bool(M == 2) | bool(M == 4):
        teorico_ber = 0.5 * erfc(np.sqrt(10**(EbN0_dBs/10)))
        teorico_ser = 1 - (1 - 0.5 * erfc(np.sqrt(10**(EbN0_dBs/10))))**np.log2(M)
    elif M == 8:
        teorico_ber = (1/3) * erfc(np.sqrt((4/10) * 10**(EbN0_dBs/10)) * np.sin(np.pi/8))
        teorico_ser = 1 - (1 - (1/3) * erfc(np.sqrt((4/10) * 10**(EbN0_dBs/10)) * np.sin(np.pi/8)))**np.log2(M)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Define o primeiro plot no subplot (BER)
    axs[0].semilogy(EbN0_dBs, teorico_ber, 'r', linewidth=2, label='Teórico')
    axs[0].semilogy(EbN0_dBs, BER, 'd', linewidth=1.5, label='Simulado')
    axs[0].set_xlabel('Eb/N0 (dB)')
    axs[0].set_ylabel('BER')
    axs[0].set_title(str(M) + '-PSK, BER vs. Eb/N0')
    axs[0].grid(True)
    axs[0].legend()

    # Define o segundo plot no subplot (SER)
    axs[1].semilogy(EbN0_dBs, teorico_ser, 'r', linewidth=2, label='Teórico')
    axs[1].semilogy(EbN0_dBs, SER, 'd', linewidth=1.5, label='Simulado')
    axs[1].set_xlabel('Eb/N0 (dB)')
    axs[1].set_ylabel('SER')
    axs[1].set_title(str(M) + '-PSK, SER vs. Eb/N0')
    axs[1].grid(True)
    axs[1].legend()

    # Exibe os plots
    plt.show()