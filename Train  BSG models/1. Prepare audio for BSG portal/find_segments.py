import sys
import re
import segment as s

def get_positions(basefile, targetspecies, datdir='wav60s', bnetdir='bnet', maxpos=60,suffix='WAV'):
    """ find start positions for all targets within ID
        ID is file id for bnet output
        targetspecies is list of species for ID
    """
    bnetfile = bnetdir + '/' + basefile + '.BirdNET.selection.table.txt'
    audiofile = datdir + '/' + basefile + '.' + suffix;

    S=s.audio2spectrogram(audiofile)
    e=s.energy(S=S)
    t,m=s.read_bnet_output(bnetfile, targetspecies)

    for species in t.keys():
        tconf=t[species]
        if len(tconf)>0:
            startpos, tpos, conf = s.score_position(tconf,m,e,maxpos=maxpos)
            #startpos is start of segment, -5 is for left margin
            print(f'{audiofile}\t{startpos-5}\t{tpos}\t{conf:.3f}\t{species}')
        else:
            print(f'{audiofile}\t-1\t-1\t0.000\t{species} NOT FOUND')
    return

def main():
    annotationfile=sys.argv[1]
    targets=s.read_targetlist(annotationfile)
    print(f'audiofile\tstartpos\ttargetpos\tconf\tspecies')
    for ID in targets.keys():
        get_positions(ID,targets[ID])

main()
