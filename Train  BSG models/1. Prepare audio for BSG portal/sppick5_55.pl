#!/usr/bin/perl

# assuming BirdNET output having 9 columns
# Selection	View	Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)	Species Code	Common Name	Confidence

$bnet_result_dir=shift;
$confth=shift;
$file=shift;

open(FD,$file);
while (<FD>) {
    ($sp,$indices)=split;
    @a=split(/,/,$indices);
    foreach $i (@a) {
	$okindex{$sp}{$i}=1;
    }
}
close(FD);

while (<>) {
    ($wavfile)=split;
    @a=split(/\//,$wavfile);
    $filename= $a[$#a];
    $filename =~ s/.WAV$//i;
    $bnetfile = ${bnet_result_dir} . "/" . $filename . ".BirdNET.selection.table.txt";
    if (open(FD,$bnetfile)) {
	<FD>;
	%spmax = ();
	%spdat = ();
	while (<FD>) {
	    s/\s+$//;
	    @a=split(/\t/);
	    $start = $a[3];
	    $end = $a[4];
	    $sp=$a[8];
	    $conf=$a[9];
	    $sp =~ s/ /_/g;
	    if ($okindex{$sp} and ($conf >= $confth) and ($start >= 5) and ($end <= 55) and ($conf > $spmax{$sp})) {
		$spmax{$sp} = $conf;
		$spdat{$sp} = "$start\t$end";
	    }
	}
	close(FD);
	#print "$bnetfile\n";
	while (($sp, $conf) = each %spmax) {
	    $count{$sp}++;
	    if ($okindex{$sp}{$count{$sp}}) {
		print "$filename\t$spdat{$sp}\t$conf\t$sp\n";
	    }
	}
    } 
}
