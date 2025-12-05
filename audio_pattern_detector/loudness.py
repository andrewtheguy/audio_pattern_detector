import warnings
from textwrap import dedent
from typing import Any
import numpy as np
import scipy.signal
from numpy.typing import NDArray

# --- From util.py ---

def valid_audio(data: NDArray[np.floating[Any]], rate: float, block_size: float) -> bool:
    """ Validate input audio data.
    
    Ensure input is numpy array of floating point data bewteen -1 and 1

    Params
    -------
    data : ndarray
        Input audio data
    rate : int
        Sampling rate of the input audio in Hz
    block_size : int
        Analysis block size in seconds 

    Returns
    -------
    valid : bool
        True if valid audio
        
    """
    if not isinstance(data, np.ndarray): # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("Data must be of type numpy.ndarray.") # pyright: ignore[reportUnreachable]
    
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError("Data must be floating point.")

    if data.ndim > 2:
        raise ValueError("Audio must be 1D or 2D.")

    if data.ndim == 2 and data.shape[1] > 5:
        raise ValueError("Audio must have five channels or less.")

    if data.shape[0] < block_size * rate:
        # Note: Users sometimes adjust block_size for short clips (e.g. if clip_seconds < 0.5)
        # to ensure this condition is met.
        raise ValueError("Audio must have length greater than the block size.")
    
    return True

# --- From iirfilter.py ---

class IIRfilter(object):
    """ IIR Filter object to pre-filtering
    
    This class allows for the generation of various IIR filters
        in order to apply different frequency weighting to audio data
        before measuring the loudness. 

    Parameters
    ----------
    G : float
        Gain of the filter in dB.
    Q : float
        Q of the filter.
    fc : float
        Center frequency of the shelf in Hz.
    rate : float
        Sampling rate in Hz.
    filter_type: str
        Shape of the filter.
    passband_gain : float
        Linear passband gain.
    """

    def __init__(self, G: float, Q: float, fc: float, rate: float, filter_type: str, passband_gain: float = 1.0):
        if Q <= 0:
            raise ValueError("Q factor must be greater than 0.")
        self.G  = G
        self.Q  = Q
        self.fc = fc
        self.rate = rate
        self.filter_type = filter_type
        self.passband_gain = passband_gain
        self._a: NDArray[np.float64] | None = None
        self._b: NDArray[np.float64] | None = None

    def __str__(self) -> str:
        filter_info = dedent("""
        ------------------------------
        type: {type}
        ------------------------------
        Gain          = {G} dB
        Q factor      = {Q} 
        Center freq.  = {fc} Hz
        Sample rate   = {rate} Hz
        Passband gain = {passband_gain}
        ------------------------------
        """.format(type = self.filter_type, 
        G=self.G, Q=self.Q, fc=self.fc, rate=self.rate,
        passband_gain=self.passband_gain))

        return filter_info

    def generate_coefficients(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """ Generates biquad filter coefficients using instance filter parameters. 
        """
        A  = 10**(self.G/40.0)
        w0 = 2.0 * np.pi * (self.fc / self.rate)
        # Avoid potential divide by zero or extreme values if Q is 0, though Q should be > 0.
        alpha = np.sin(w0) / (2.0 * self.Q)

        if self.filter_type == 'high_shelf':
            b0 =      A * ( (A+1) + (A-1) * np.cos(w0) + 2 * np.sqrt(A) * alpha )
            b1 = -2 * A * ( (A-1) + (A+1) * np.cos(w0)                          )
            b2 =      A * ( (A+1) + (A-1) * np.cos(w0) - 2 * np.sqrt(A) * alpha )
            a0 =            (A+1) - (A-1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 =      2 * ( (A-1) - (A+1) * np.cos(w0)                          )
            a2 =            (A+1) - (A-1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        elif self.filter_type == 'low_shelf':
            b0 =      A * ( (A+1) - (A-1) * np.cos(w0) + 2 * np.sqrt(A) * alpha )
            b1 =  2 * A * ( (A-1) - (A+1) * np.cos(w0)                          )
            b2 =      A * ( (A+1) - (A-1) * np.cos(w0) - 2 * np.sqrt(A) * alpha )
            a0 =            (A+1) + (A-1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 =     -2 * ( (A-1) + (A+1) * np.cos(w0)                          )
            a2 =            (A+1) + (A-1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        elif self.filter_type == 'high_pass':
            b0 =  (1 + np.cos(w0))/2
            b1 = -(1 + np.cos(w0))
            b2 =  (1 + np.cos(w0))/2
            a0 =   1 + alpha
            a1 =  -2 * np.cos(w0)
            a2 =   1 - alpha
        elif self.filter_type == 'low_pass':
            b0 =  (1 - np.cos(w0))/2
            b1 =  (1 - np.cos(w0))
            b2 =  (1 - np.cos(w0))/2
            a0 =   1 + alpha
            a1 =  -2 * np.cos(w0)
            a2 =   1 - alpha
        elif self.filter_type == 'peaking':
            b0 =   1 + alpha * A
            b1 =  -2 * np.cos(w0)
            b2 =   1 - alpha * A
            a0 =   1 + alpha / A
            a1 =  -2 * np.cos(w0)
            a2 =   1 - alpha / A
        elif self.filter_type == 'notch':
            b0 =   1 
            b1 =  -2 * np.cos(w0)
            b2 =   1
            a0 =   1 + alpha
            a1 =  -2 * np.cos(w0)
            a2 =   1 - alpha
        elif self.filter_type == 'high_shelf_DeMan':
            K_tan  = np.tan(np.pi * self.fc / self.rate) 
            Vh = np.power(10.0, self.G / 20.0)
            Vb = np.power(Vh, 0.499666774155)
            a0_ = 1.0 + K_tan / self.Q + K_tan * K_tan
            b0 = (Vh + Vb * K_tan / self.Q + K_tan * K_tan) / a0_
            b1 =  2.0 * (K_tan * K_tan -  Vh) / a0_
            b2 = (Vh - Vb * K_tan / self.Q + K_tan * K_tan) / a0_
            a0 =  1.0
            a1 =  2.0 * (K_tan * K_tan - 1.0) / a0_
            a2 = (1.0 - K_tan / self.Q + K_tan * K_tan) / a0_
        elif self.filter_type == 'high_pass_DeMan':
            K_tan  = np.tan(np.pi * self.fc / self.rate)
            a0 =  1.0
            a1 =  2.0 * (K_tan * K_tan - 1.0) / (1.0 + K_tan / self.Q + K_tan * K_tan)
            a2 = (1.0 - K_tan / self.Q + K_tan * K_tan) / (1.0 + K_tan / self.Q + K_tan * K_tan)
            b0 =  1.0
            b1 = -2.0
            b2 =  1.0
        else:
            raise ValueError("Invalid filter type", self.filter_type)            

        return np.array([b0, b1, b2])/a0, np.array([a0, a1, a2])/a0

    def apply_filter(self, data: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """ Apply the IIR filter to an input signal.
        """
        # scipy.signal.lfilter returns an array.
        # We check types explicitly or ignore.
        result = scipy.signal.lfilter(self.b, self.a, data) # type: ignore
        return self.passband_gain * np.asarray(result).astype(np.float64)

    @property
    def a(self) -> NDArray[np.float64]:
        if self._a is None:
            self._b, self._a = self.generate_coefficients()
        return self._a # type: ignore

    @property
    def b(self) -> NDArray[np.float64]:
        if self._b is None:
            self._b, self._a = self.generate_coefficients()
        return self._b # type: ignore


# --- From meter.py ---

class Meter(object):
    """ Meter object which defines how the meter operates

    Defaults to the algorithm defined in ITU-R BS.1770-4.

    Parameters
    ----------
    rate : float
        Sampling rate in Hz.
    filter_class : str
        Class of weigthing filter used.
        - 'K-weighting'
        - 'Fenton/Lee 1'
        - 'Fenton/Lee 2'
        - 'Dash et al.'
        - 'DeMan'
    block_size : float
        Gating block size in seconds.
    """

    def __init__(self, rate: float, filter_class: str ="K-weighting", block_size: float = 0.400):
        self.rate = rate
        self.block_size = block_size
        self._filter_class: str | None = None
        self._filters: dict[str, IIRfilter] = {}
        # Set property (initializes filters)
        self.filter_class = filter_class

    def integrated_loudness(self, data: NDArray[np.floating[Any]]) -> float:
        """ Measure the integrated gated loudness of a signal.
        """
        input_data = data.copy()
        valid_audio(input_data, self.rate, self.block_size)

        if input_data.ndim == 1:
            input_data = np.reshape(input_data, (input_data.shape[0], 1))

        numChannels = input_data.shape[1]        
        numSamples  = input_data.shape[0]

        # Apply frequency weighting filters - account for the acoustic respose of the head and auditory system
        for _, filter_stage in self._filters.items():
            for ch in range(numChannels):
                input_data[:,ch] = filter_stage.apply_filter(input_data[:,ch])

        G = [1.0, 1.0, 1.0, 1.41, 1.41] # channel gains
        T_g = self.block_size # 400 ms gating block standard
        Gamma_a = -70.0 # -70 LKFS = absolute loudness threshold
        overlap = 0.75 # overlap of 75% of the block duration
        step = 1.0 - overlap # step size by percentage

        T = numSamples / self.rate # length of the input in seconds
        
        # Check against small files to prevent zero blocks
        if T < T_g:
             # Just use one block if simpler, or let it fail?
             # The loop logic:
             pass

        numBlocks = int(np.round(((T - T_g) / (T_g * step)))+1) # total number of gated blocks (see end of eq. 3)
        if numBlocks < 1:
            # If strictly following equation, numBlocks might be < 1 if T ~ T_g but slightly less due to float/rounding?
            # Or if T < T_g. 'valid_audio' checks T >= T_g but if they are equal floating point issues might occur.
            numBlocks = 1
            
        j_range = np.arange(0, numBlocks) # indexed list of total blocks
        z = np.zeros(shape=(numChannels,numBlocks)) # instantiate array - trasponse of input

        for i in range(numChannels): # iterate over input channels
            for j in j_range: # iterate over total frames
                l_bound = int(T_g * (j * step    ) * self.rate) # lower bound of integration (in samples)
                u_bound = int(T_g * (j * step + 1) * self.rate) # upper bound of integration (in samples)
                
                # Check boundaries
                if u_bound > numSamples:
                    u_bound = numSamples
                if l_bound >= u_bound:
                    continue # Should not happen if numBlocks calc is correct but safety
                
                # caluate mean square of the filtered for each block (see eq. 1)
                # Ensure we don't divide by zero if u-l is small, though expected to be T_g * rate
                segment = input_data[l_bound:u_bound, i]
                if len(segment) > 0:
                    z[i,j] = np.mean(np.square(segment))
                else:
                    z[i,j] = 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # loudness for each jth block (see eq. 4)
            # Avoid log10(0)
            # z is (numChannels, numBlocks). We want to sum weighted channels for each block.
            # G is list of gains.
            # We can compute weighted sum across axis 0 (channels)
            weighted_z_list = [G[i] * z[i] for i in range(numChannels)]
            z_sum_any = np.sum(weighted_z_list, axis=0)
            
            # Cast to ensure type checker knows it's numeric/float
            if np.isscalar(z_sum_any):
                z_sum = float(z_sum_any) # pyright: ignore[reportArgumentType]
            else:
                z_sum = np.asarray(z_sum_any, dtype=np.float64)

            # Handle empty or zero blocks
            if np.isscalar(z_sum):
                if z_sum <= 0: # pyright: ignore[reportOperatorIssue]
                    loudness_blocks = np.array([-np.inf])
                else:
                    loudness_blocks = np.array([-0.691 + 10.0 * np.log10(z_sum)])
            else:
                 l_list = []
                 # Assert it is array for type checker
                 assert isinstance(z_sum, np.ndarray)
                 for val in z_sum:
                     if val <= 0:
                         l_list.append(-np.inf)
                     else:
                         l_list.append(-0.691 + 10.0 * np.log10(val))
                 loudness_blocks = np.array(l_list)

        # find gating block indices above absolute threshold
        J_g = [j for j,l_j in enumerate(loudness_blocks) if l_j >= Gamma_a]
        
        if not J_g:
             return -np.inf # Silence

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) # type: ignore
            # calculate the average of z[i,j] as show in eq. 5
            # z is shape (channels, blocks)
            z_avg_gated = [np.mean([z[i,j] for j in J_g]) for i in range(numChannels)]
            
        # calculate the relative threshold value (see eq. 6)
        z_avg_sum = np.sum([G[i] * z_avg_gated[i] for i in range(numChannels)])
        if z_avg_sum <= 0:
            Gamma_r = -np.inf
        else:
            Gamma_r = -0.691 + 10.0 * np.log10(z_avg_sum) - 10.0

        # find gating block indices above relative and absolute thresholds  (end of eq. 7)
        J_g = [j for j,l_j in enumerate(loudness_blocks) if (l_j > Gamma_r and l_j > Gamma_a)]
        
        if not J_g:
            return -np.inf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) # type: ignore
            # calculate the average of z[i,j] as show in eq. 7 with blocks above both thresholds
            z_avg_gated = np.nan_to_num(np.array([np.mean([z[i,j] for j in J_g]) for i in range(numChannels)]))
        
        # calculate final loudness gated loudness (see eq. 7)
        with np.errstate(divide='ignore'):
             z_final_sum = np.sum([G[i] * z_avg_gated[i] for i in range(numChannels)])
             if z_final_sum <= 0:
                 lufs = -np.inf
             else:
                 lufs = -0.691 + 10.0 * np.log10(z_final_sum)

        return float(lufs)
    
    @property
    def filter_class(self) -> str:
        if self._filter_class is None:
             raise ValueError("Filter class is not set")
        return self._filter_class

    @filter_class.setter
    def filter_class(self, value: str):
        self._filters = {} # reset (clear) filters
        self._filter_class = value
        if   self._filter_class == "K-weighting":
            self._filters['high_shelf'] = IIRfilter(4.0, 1/np.sqrt(2), 1500.0, self.rate, 'high_shelf')
            self._filters['high_pass'] = IIRfilter(0.0, 0.5, 38.0, self.rate, 'high_pass')
        elif self._filter_class == "Fenton/Lee 1":
            self._filters['high_shelf'] = IIRfilter(5.0, 1/np.sqrt(2), 1500.0, self.rate, 'high_shelf')
            self._filters['high_pass'] = IIRfilter(0.0, 0.5, 130.0, self.rate, 'high_pass')
            self._filters['peaking'] = IIRfilter(0.0, 1/np.sqrt(2), 500.0, self.rate, 'peaking')
        elif self._filter_class == "Fenton/Lee 2": # not yet implemented 
            self._filters['high_shelf'] = IIRfilter(4.0, 1/np.sqrt(2), 1500.0, self.rate, 'high_shelf')
            self._filters['high_pass'] = IIRfilter(0.0, 0.5, 38.0, self.rate, 'high_pass')
        elif self._filter_class == "Dash et al.":
            self._filters['high_pass'] = IIRfilter(0.0, 0.375, 149.0, self.rate, 'high_pass')
            self._filters['peaking'] = IIRfilter(-2.93820927, 1.68878655, 1000.0, self.rate, 'peaking')
        elif self._filter_class == "DeMan":
            self._filters['high_shelf_DeMan'] = IIRfilter(3.99984385397, 0.7071752369554193, 1681.9744509555319, self.rate, 'high_shelf_DeMan')
            self._filters['high_pass_DeMan'] = IIRfilter(0.0, 0.5003270373253953, 38.13547087613982, self.rate, 'high_pass_DeMan')
        elif self._filter_class == "custom":
            pass
        else:
            raise ValueError(f"Invalid filter class: {self._filter_class}")


# --- From normalize.py ---

class normalize:
    """Namespace for normalization functions."""
    
    @staticmethod
    def peak(data: NDArray[np.floating[Any]], target: float) -> NDArray[np.floating[Any]]:
        """ Peak normalize a signal.
        
        Normalize an input signal to a user specifed peak amplitude.   

        Params
        -------
        data : ndarray
            Input multichannel audio data.
        target : float
            Desired peak amplitude in dB.

        Returns
        -------
        output : ndarray
            Peak normalized output data.
        """
        # find the amplitude of the largest peak
        current_peak = np.max(np.abs(data))

        # calculate the gain needed to scale to the desired peak level
        gain = np.power(10.0, target/20.0) / current_peak
        output = gain * data
        
        # check for potentially clipped samples
        if np.max(np.abs(output)) >= 1.0:
            warnings.warn("Possible clipped samples in output.")

        return output

    @staticmethod
    def loudness(data: NDArray[np.floating[Any]], input_loudness: float, target_loudness: float) -> NDArray[np.floating[Any]]:
        """ Loudness normalize a signal.
        
        Normalize an input signal to a user loudness in dB LKFS.   

        Params
        -------
        data : ndarray
            Input multichannel audio data.
        input_loudness : float
            Loudness of the input in dB LUFS. 
        target_loudness : float
            Target loudness of the output in dB LUFS.
            
        Returns
        -------
        output : ndarray
            Loudness normalized output data.
        """    
        # calculate the gain needed to scale to the desired loudness level
        # Guard against -inf input_loudness (silence)
        if np.isneginf(input_loudness):
            gain = 0.0 # Silence remains silence
        else:
            delta_loudness = target_loudness - input_loudness
            gain = np.power(10.0, delta_loudness/20.0)

        output = gain * data

        # check for potentially clipped samples
        if np.max(np.abs(output)) >= 1.0:
            warnings.warn("Possible clipped samples in output.")

        return output
