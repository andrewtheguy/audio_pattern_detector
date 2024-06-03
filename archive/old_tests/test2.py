
if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt
    from dtw import dtw

    #Loading audio files
    y1, sr1 = librosa.load(librosa.ex('pistachio'))
    y2, sr2 = librosa.load(librosa.ex('pistachio'), offset=10)

    #Showing multiple plots using subplot
    plt.subplot(1, 2, 1)
    mfcc1 = librosa.feature.mfcc(y=y1,sr=sr1)   #Computing MFCC values
    librosa.display.specshow(mfcc1)

    plt.subplot(1, 2, 2)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
    librosa.display.specshow(mfcc2)

    alignment = dtw(mfcc1.T, mfcc2.T,keep_internals=True)
    dist=alignment.normalizedDistance
    print("The normalized distance between the two : ",dist)   # 0 for similar audios

    plt.imshow(alignment.costMatrix, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
    plt.plot(alignment.index1, alignment.index2, 'w')   #creating plot for DTW

    plt.show()  #To display the plots graphically