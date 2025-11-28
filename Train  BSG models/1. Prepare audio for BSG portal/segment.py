import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def read_targetlist(filename):
    """ filename is text file with fileid,start,end,conf,species fields separated by tab
        there are no header lines
    """
    f = open(filename,"r")
    targets={}
    for line in f:
        foo = line.rstrip().split("\t")
        fileid=foo[0]
        species=foo[4]
        if fileid in targets:
            targets[fileid].append(species)
        else:
            targets[fileid]=[species]
    f.close()
    return targets

def audio2spectrogram(audiofile):
    """ returns magnitude spectrum """
    sr=48000
    hop_length=int(sr/100)

    # kaiser_fast is 4x faster than kaiser_best
    y,sr=librosa.load(audiofile, sr=sr, res_type='kaiser_fast')
    S=np.abs(librosa.stft(y, n_fft=1024, hop_length=hop_length))
    return S

def plot_audio(S,e,conf,bf=0,ef=None,maxfreq=None):
    """ plot spectrogram and two vectors: energy (scaled for max 1) and conf 
        bf and ef and in seconds
    """
    if ef is None:
        ef=len(e)
            
    sr=48000
    hop_length=int(sr/100)

    if maxfreq is None:
        maxfreq=sr/2

    fig, ax = plt.subplots(nrows=2, ncols=1)
    img=librosa.display.specshow(librosa.amplitude_to_db(S[:,bf*100:ef*100],ref=np.max),sr=sr,hop_length=hop_length, y_axis='linear',x_axis='time', ax=ax[0])
    ax[0].set_ylim(0,maxfreq)
    t=np.arange(bf,ef+1)
    x=e[bf:ef]
    x=x/max(x)
    ax[1].step(t,np.concatenate(((x[0],), x)) )
    x=conf[bf:ef]
    ax[1].step(t,np.concatenate(((x[0],), x)) )
    ax[1].set_xlim(bf,ef)
    ax[1].set_ylim(-.05,1.05)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

def energy(audiofile=None, S=None, skip_bins=3):
    """ computes energy along time with 1-second resolution 
        excluding frequencies below skip_bins
    """
    sr=48000
    hop_length=int(sr/100)

    # kaiser_fast is 4x faster than kaiser_best
    if audiofile is not None:
        y,sr=librosa.load(audiofile, sr=sr, res_type='kaiser_fast')
        S=np.abs(librosa.stft(y, n_fft=1024, hop_length=hop_length))

    e=np.sum(S[skip_bins:,:]**2,axis=0)

    # create energy vector with time resolution of 1 second ==> merge 100 frames
    num_merge_frames=100
    me=np.zeros(int(np.ceil(len(e)/num_merge_frames)))
    for i in range(len(me)):
        starti=i*num_merge_frames
        endi=min((i+1)*num_merge_frames,len(e))
        me[i] = np.mean(e[starti:endi])

    return me

def read_bnet_output(filename, targetlist):
    """ filename is bnet output and targetlist is list of target species
        function returns dictionary for each species with time as key and conf as value
        and maxconf that contains maximum conf for each time among any species
        including those not in the targetlist
    """
    f = open(filename,"r")
    next(f)
    targetconf={x:{} for x in targetlist}
    maxconf={}
    for line in f:
        #start,end,sci,common,conf_str = line.rstrip().split(";")
        # read v2.4 format
        sel,view,channel,start,end,lowf,highf,sci,common,conf_str = line.rstrip().split("\t")
        spname = common.replace(" ","_")
        conf = float(conf_str)
        if spname in targetconf:
            targetconf[spname][start]=conf
        if start in maxconf:
            if conf > maxconf[start]:
                maxconf[start]=conf
        else:
                maxconf[start]=conf

    f.close()
    return targetconf, maxconf

def dict2vec(d, len=61):
    """ convert dictionary d['pos']=val into val vector where pos=0..len """
    v=np.zeros(len)
    #for str_pos in sorted(d.keys(), key=lambda item: float(item)):
    for str_pos in d.keys():
        i=int(float(str_pos))
        v[i] = d[str_pos]
    return v

def score_position(tconf, aconf, e, maxpos=60, seg_len=10, pos_margin=5, confr_th=0.9, conf_weight=1, costr_th=0.9, info=0):
    """ score each position to get (pos_margin+seg_len+pos_margin)-second segment
        tconf: dictionary[pos]=conf for target species
        aconf: dictionary[pos]=conf for any species
        e: array energy for each position
        confr_th: positions to be accepted as candidates, conf > maxconf*confr_th
        conf_weight: weight of aconf when summing aconf and energy (cost function to be minimized)
        costr_th: favor center positions, accept if cost*costr_th < mincost
    """

    a=dict2vec(aconf, maxpos+1)
    
    # bnet analysis window length in seconds
    bnet_len=3
    # take max conf within pos margins
    # maxi_str = max(tconf,key=tconf.get)
    # maxconf = tconf[maxi_str]
    # maxi = int(float(maxi_str))
    maxi_str=-1
    maxconf = 0
    for pos_str, conf in tconf.items():        
        pos=int(float(pos_str))
        if pos_margin <= pos <= maxpos-(bnet_len+pos_margin):
            if conf > maxconf:
                maxconf=conf
                maxi = pos
        
    ok=0

    best_posdif=2*seg_len
    best_cost=100
    best_conf=0
    best_start_pos=-1
    best_target_pos=-1

    # for str_pos,conf in tconf.items():
    for pos_str in sorted(tconf.keys(), key=lambda item: float(item)):
        pos=int(float(pos_str))
        if pos_margin <= pos <= maxpos-(bnet_len+pos_margin):
            if tconf[pos_str] > confr_th * maxconf:
                # loop over candidate positions, i is start index for seg_len-segment
                start_i = max(pos_margin, pos-(seg_len-bnet_len))
                end_i = min(pos, maxpos - (seg_len + pos_margin))
                if info:
                    print(f'ok pos {pos_str}, range {start_i}..{end_i}: {tconf[pos_str]:.3f} (max {maxconf:.3f})')
                # pos10 = 0..(seg_len-bnet_len)
                mincost=100
                min_pos_dif=2*seg_len
                for i in range(start_i, end_i+1):
                    x=e[i:i+seg_len]
                    x=x/max(x)
                    # sum energy and aconf at the beginning and end of segment after begin_gap and before end_gap
                    cost=x[0]+x[seg_len-1] + conf_weight*(a[i]+a[i+seg_len-1])
                    if cost < mincost:
                        mincost=cost
                        mini=i

                for i in range(start_i, end_i+1):
                    x=e[i:i+seg_len]
                    x=x/max(x)
                    # sum energy and aconf at the beginning and end of segment after begin_gap and before end_gap
                    cost=x[0]+x[seg_len-1] + conf_weight*(a[i]+a[i+seg_len-1])                
                    pos10 = i-start_i
                    # if accepted candidate, choose one with minimum pos_dif (position-central_segment)
                    pos_dif=abs(pos-i - (seg_len-bnet_len)/2)
                    if info>1:
                        print(f'  pos {i} {pos10}: e: {x[0]:.3f} {x[seg_len-1]:.3f} aconf: {a[i]:.3f} {a[i+seg_len-1]:.3f} cost {cost:.3f} ({mincost/cost:.3f}) posdif {pos_dif:.2f}')
                    if cost*costr_th < mincost and pos_dif < min_pos_dif:
                        min_pos_dif=pos_dif
                        besti=i
                        bestcost = cost

                if min_pos_dif < best_posdif:
                     best_posdif = min_pos_dif
                     best_cost = bestcost
                     best_conf = tconf[pos_str]
                     best_start_pos = besti
                     best_target_pos = pos

                if info:
                    print(f'  mincost {mincost:.3f} at pos {mini} min_posdif {min_pos_dif} besti {besti}')
    if info:
        print(f'best start {best_start_pos} targetpos {best_target_pos} conf {best_conf:.3f} cost {best_cost:.3f} posdif {best_posdif:.2f}')
    return best_start_pos, best_target_pos, best_conf

def get_positions(ID, targetspecies, datdir='data', bnetdir='data', suffix='WAV', maxpos=60):
    """ find start positions for all targets within ID
        ID is file id
        targetspecies is list of species for ID
    """
    audiofile= datdir + '/' + ID + '.' + suffix
    bnetfile= bnetdir + '/' + ID + '.bnet'

    S=audio2spectrogram(audiofile)
    e=energy(S=S)
    t,m=read_bnet_output(bnetfile, targetspecies)

    for species in t.keys():
        tconf=t[species]
        startpos, tpos, conf = score_position(tconf,m,e,maxpos=maxpos)
        print(f'{species}: startpos {startpos} speciespos {tpos} conf {conf:.3f}')

    return
