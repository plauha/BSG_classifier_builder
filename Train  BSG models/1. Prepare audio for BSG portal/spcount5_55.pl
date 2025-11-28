#!/usr/bin/perl

# assuming BirdNET output having 9 columns
# Selection	View	Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)	Species Code	Common Name	Confidence

$bnet_result_dir=shift;

while (<>) {
    ($wavfile)=split;
    @a=split(/\//,$wavfile);
    $filename= $a[$#a];
    $filename =~ s/.WAV$//i;
    $bnetfile = ${bnet_result_dir} . "/" . $filename . ".BirdNET.selection.table.txt";
    if (open(FD,$bnetfile)) {
	<FD>;
	%spmax = ();
	while (<FD>) {
	    s/\s+$//;
	    @a=split(/\t/);
	    $begtime=$a[3];
	    $endtime=$a[4];
	    if ($begtime >= 5 & $endtime <= 55) {
		$sp=$a[8];
		$conf=$a[9];
		$sp =~ s/ /_/g;
		$spmax{$sp} = $conf if ($conf > $spmax{$sp});
	    }
	}
	close(FD);
	#print "$bnetfile\n";
	while (($sp, $conf) = each %spmax) {
	    $spon{$sp}=1;
	    if ($conf > 0.9) {$count9{$sp}++;}
	    if ($conf > 0.7) {$count7{$sp}++;}
	    if ($conf > 0.5) {$count5{$sp}++;}
	    if ($conf > 0.3) {$count3{$sp}++;}
	    if ($conf > 0.1) {$count1{$sp}++;}
	}
    } 
}

foreach $sp (sort keys %spon) {
    $c1=0;$c3=0;$c5=0;$c7=0;$c9=0;
    $c1 = $count1{$sp} if (exists($count1{$sp}));
    $c3 = $count3{$sp} if (exists($count3{$sp}));
    $c5 = $count5{$sp} if (exists($count5{$sp}));
    $c7 = $count7{$sp} if (exists($count7{$sp}));
    $c9 = $count9{$sp} if (exists($count9{$sp}));
    print "$sp\t$c1\t$c3\t$c5\t$c7\t$c9\n";
}
