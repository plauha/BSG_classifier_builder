#!/usr/bin/perl

# in puhti: module load ffmpeg

$seglen=20;
$odir=shift;

while (<>) {
    s/\s+$//;
    ($ifile, $start)=split();
    $start=int($start);
    $min = sprintf("%02d",int($start/60));
    $sec = sprintf("%02d",$start - 60*$min);
    $ss = "00:$min:$sec";
    $dur = "00:00:$seglen";

    @a=split(/\//,$ifile);
    $a[$#a] =~ s/.WAV//;
    $fileid=join('_', @a[1..$#a]);

    $ofile = "$odir/$fileid" . "_$start" . ".mp3";
    
    $cmd = "ffmpeg -nostdin -loglevel error -ss $ss -i $ifile -t $dur -ac 1 -ar 32k -codec:a libmp3lame -aq 2 $ofile";
    system($cmd);
    
}
