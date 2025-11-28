#!/usr/bin/perl

$ofile=shift;
$info=shift;

<>;
while (<>) {
    s/\s+$//;
    ($audiofile,$startpos,$targetpos,$conf,$species)=split(/\t/);
    if ($startpos >= 0) {
	push(@{$start_dat{$audiofile}}, $startpos);
	push(@{$tpos_dat{$audiofile}}, $targetpos);
	push(@{$conf_dat{$audiofile}}, $conf);
	push(@{$species_dat{$audiofile}}, $species);
    }
}

# my @sorted_indexes = sort { $unsorted[$b] <=> $unsorted[$a] } 0..$#unsorted;

$linemark="+-----------------------------------------------------------\n";

open(OFD,">",$ofile) or die "ERROR: cannot write to '$ofile'. $!\n";

foreach $audiofile (sort keys %start_dat) {
    @arr=@{$start_dat{$audiofile}};
    @brr=@{$tpos_dat{$audiofile}};
    # sort first based on start pos and if same then based on target pos
    @ind = sort {$arr[$a] <=> $arr[$b] || $brr[$a] <=> $brr[$b]} 0..$#arr;
    @start = @arr[@ind];
    @tpos=@{$tpos_dat{$audiofile}}[@ind];
    @conf=@{$conf_dat{$audiofile}}[@ind];
    @species=@{$species_dat{$audiofile}}[@ind];

    $s=join(' ',@start);
    $t=join(',',@tpos);
    $c=join(',',@conf);
    $sp=join(',',@species);
    print "$audiofile $s ($t) ($c) ($sp)\n" if ($info);

    $n=scalar(@start);    
    $nsegs=0;
    $prev=-1;
    $prev_min_tpos=-1;

    @ss=();

    for($i=0; $i<$n; $i++) {
	if ($prev == -1) {
	    $prev=$start[$i];
	    $prev_min_tpos=$tpos[$i];
	    push(@ss, $prev);
	    $nsegs++;
	    print $linemark if ($info);
	} else {
	    $prev_min_tpos = $tpos[$i] if ($tpos[$i] < $prev_min_tpos);
	    if ($tpos[$i] > $prev+5+7) {
		# try if previous tpos can fit into this window
		if ($prev_min_tpos >= $start[$i]+5) {
		    $ss[$nsegs-1] = $start[$i];
		    $prev=$start[$i];
		    print "| can be merged!\n" if ($info);
		} else {
		    $prev_min_tpos=$tpos[$i];
		    $prev=$start[$i];
		    push(@ss, $prev);
		    $nsegs++;
		    print $linemark if ($info);
		}
	    }
	}
	print "|    $start[$i] $tpos[$i] $conf[$i] $species[$i]\n" if ($info);
    }
    if ($info) {
	print $linemark;
	print "$n detections ==> $nsegs segments:\n";    
	foreach $spos (@ss) {
	    print "  start_pos $spos\n";

	}
	print "\n";
    }

    foreach $spos (@ss) {
	print OFD "$audiofile $spos\n";
    }


    $nouts += $nsegs;
}

close(OFD);

print "$nouts total segments\n";
