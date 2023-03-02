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

        info = np.random.randint(0, 4, num_symbols)

        """
        Mapeamento Gray 4-PSK (QPSK):
        
            - Bit 00 = Símbolo -1+1j (portadora com fase phi = +135)
            - Bit 01 = Símbolo +1+1j (portadora com fase phi = +45)
            - Bit 11 = Símbolo +1-1j (portadora com fase phi = -45)
            - Bit 10 = Símbolo -1-1j (portadora com fase phi = -135)
        """
        constelacao = np.array([1+0j, 0+1j, -1+0j, 0-1j])

        # Sinal de informação a ser transmitido
        bits = np.array([[0,1],[0,0],[1,0],[1,1]])
        x = bits[info].reshape(-1)
        
        # Sinal transmitido através da modulação do vetor de bits de informação
        s = constelacao[info]
        
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
            
            if (bool(phi[j] >= 0) & bool(phi[j] <= 45)) | bool(phi[j] > 315): # Símbolo 1 -> 01 -> 1+0j
                b1 = 0
                b2 = 1
                c = complex(b1,b2)
            elif (bool(phi[j] > 45) & bool(phi[j] < 135)): # Símbolo 2 -> 00 -> 0+j1
                b1 = 0
                b2 = 0
                c = complex(b1,b2)
            elif (bool(phi[j] > 135) & bool(phi[j] < 225)): # Símbolo 3 -> 10 -> -1+j0
                b1 = 1
                b2 = 0
                c = complex(b1,b2)
            elif (bool(phi[j] > 225) & bool(phi[j] < 315)): # Símbolo 4 -> 11 -> 0-j1
                b1 = 1
                b2 = 1
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

    return BER, SER, r


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

        info = np.random.randint(0, 2, num_bits)

        """
        Mapeamento 2-PSK (BPSK):
        
            - Bit 0 = Símbolo -1 (portadora com fase phi = 0)
            - Bit 1 = Símbolo +1 (portadora com fase phi = 180)
        """
        constelacao = np.array([-1, 1])

        # Sinal de informação a ser transmitido
        bits = np.array([0,1])
        x = bits[info].reshape(-1)
        
        # Sinal transmitido através da modulação do vetor de bits de informação
        s = constelacao[info]

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
    
    return BER, SER, r

def PlotConstellation(num_bits, EbN0_range, M):
    """Faz o plot do diagrama de constelação para uma modulação PSK de ordem M

    Args:
        num_bits (integer): Número de bits de informação a serem utilizados na simulação.
        EbN0_range (list): Lista das 3 SNRs a serem plotadas. 
        M (integer): Ordem da modulação PSK
    """

    if M == 2:
        recebidos = []
        for i in range(len(EbN0_range)):

            ber,ser,r = bpsk(num_bits, [EbN0_range[i]])

            # downsample em 1e4 vezes para ajudar na visualização do efeito da SNR
            r = r[::10000]

            # cria um array com valores y iguais a zero
            y = np.zeros_like(r)

            recebidos.append(r)

        # cria um array booleano para identificar os valores x positivos
        mask0 = recebidos[0] > 0
        mask1 = recebidos[1] > 0
        mask2 = recebidos[2] > 0

        # Cria figura e subplots
        fig, axs = plt.subplots(1, 3, figsize=(14,4), sharex=False)

        axs[0].scatter(recebidos[0][mask0], y[mask0], color='red')
        axs[0].scatter(recebidos[0][~mask0], y[~mask0], color='blue')
        axs[0].set_ylabel('Q')
        axs[0].set_xlabel('I')
        axs[0].set_title('Eb/N0 = ' + str(EbN0_range[0]) + ' dB')

        axs[1].scatter(recebidos[1][mask1], y[mask1], color='red')
        axs[1].scatter(recebidos[1][~mask1], y[~mask1], color='blue')
        axs[1].set_ylabel('Q')
        axs[1].set_xlabel('I')
        axs[1].set_title('Eb/N0 = ' + str(EbN0_range[1]) + ' dB')

        axs[2].scatter(recebidos[2][mask2], y[mask2], color='red')
        axs[2].scatter(recebidos[2][~mask2], y[~mask2], color='blue')
        axs[2].set_ylabel('Q')
        axs[2].set_xlabel('I')
        axs[2].set_title('Eb/N0 = ' + str(EbN0_range[2]) + ' dB')

        plt.subplots_adjust(wspace=0.4)
        plt.show()
    

def PlotBER_SER(num_bits, EbN0_dBs, M):
    """Faz o plot das curvas de BER e SER simuladas e teóricas, tendo como entrada os valores da simulação.

    Args:
        EbN0_dBs (array): Array de medidas da relação entre a energia média transmitida por bit e a densidade espectral de potência do ruído aditivo branco gaussiano (AWGN) no canal de comunicação
        BER (array): Taxa de erro de bit simulada
        SER (array): Taxa de erro de símbolo simulada
        M (integer): Ordem da modulação. Ex.: BPSK -> M = 2, 2-PSK
    """    
    
    if M == 2:
        teorico_ber = 0.5 * erfc(np.sqrt(10**(EbN0_dBs/10)))
        teorico_ser = 1 - (1 - 0.5 * erfc(np.sqrt(10**(EbN0_dBs/10))))**np.log2(M)
        BER, SER, r = bpsk(num_bits, EbN0_dBs)
    elif M == 4:
        teorico_ber = 0.5 * erfc(np.sqrt(10**(EbN0_dBs/10)))
        teorico_ser = 1 - (1 - 0.5 * erfc(np.sqrt(10**(EbN0_dBs/10))))**np.log2(M)
        BER, SER, r = qpsk(num_bits, EbN0_dBs)
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


def PlotWaveforms(bits_seq, fc, EbN0):
    """Plota as etapas da modulação e transmissão BPSK em banda passante

    Args:
        bits_seq (array): Sequência de bits de informação em banda base
        fc (integer): Frequência do sinal da portadora em Hertz
        EbN0 (floar): SNR da transmissão. Quanto mais baixa, pior o canal AWGN 
    """

    # Sinal transmitido obtido através da modulação do vetor de bits de informação
    bpsk_seq = np.round(2*bits_seq-1)

    # Duração de cada período da portadora em segundos
    Tc = 1/fc

    # Tempo de amostragem em segundos
    Ts = Tc/100

    # Número de amostras por período da portadora
    N = int(Tc/Ts)

    # Período da portadora correspondente a -1
    cos_neg = np.cos(2*np.pi*fc*np.arange(N)*Ts)

    # Período da portadora correspondente a 1
    cos_pos = np.cos(2*np.pi*fc*np.arange(N)*Ts + np.pi)

    # Sequência de símbolos modulada
    bpsk_mod = np.concatenate([cos_neg if s==-1 else cos_pos for s in bpsk_seq])

    # Eixo do tempo
    t = np.arange(len(bpsk_mod))*Ts

    # Variância do sinal transmitido e adição de ruído AWGN
    var = (1/np.sqrt(2))*10**(-EbN0/20)
    noise = np.random.normal(0, var, len(t))

    # Cria figura e subplots
    fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=False)

    # Plota no primeiro subplot
    axs[0].step(np.arange(0,len(bits_seq)),np.array(bits_seq))
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Sinal de Informação em Banda Base (duração bit = 1 s)')

    axs[1].step(np.arange(0,len(bpsk_seq)),np.array(bpsk_seq))
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Bits mapeados para símbolos BPSK')

    # Plota no segundo subplot
    axs[2].plot(t, bpsk_mod)
    axs[2].set_title('Sinal Modulado Transmitido em Banda Passante ($f_c$ = ' + str(fc/1000) + ' kHz)')
    axs[2].set_ylabel('Amplitude')

    # Plota no terceiro subplot
    axs[3].plot(t, noise+bpsk_mod)
    axs[3].set_title('Sinal Ruidoso Recebido ($f_c$ = ' + str(fc/1000) + ' kHz, EbN0 = ' + str(EbN0) + ' dB)')
    axs[3].set_xlabel('Tempo (s)')
    axs[3].set_ylabel('Amplitude')


    plt.subplots_adjust(hspace=0.8)

    # Exibe o plot
    plt.show()