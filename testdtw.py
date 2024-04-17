import librosa
from matplotlib.pylab import norm
import matplotlib.pyplot as plt
from dtw import dtw

#Loading audio files
y1, sr1 = librosa.load('./audio_clips/rthk_beep.wav') 
y2, sr2 = librosa.load('./tmp/inputs/happydailytest.wav') 

#Showing multiple plots using subplot
plt.subplot(1, 2, 1) 
mfcc1 = librosa.feature.mfcc(y=y1,sr=sr1)   #Computing MFCC values
librosa.display.specshow(mfcc1)

plt.subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
librosa.display.specshow(mfcc2)

d, cost_matrix, acc_cost_matrix, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print("The normalized distance between the two : ",d)   # 0 for similar audios 

plt.imshow(cost_matrix.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
plt.plot(path[0], path[1], 'w')   #creating plot for DTW

plt.savefig(f'./tmp/dtw.png')
plt.close()