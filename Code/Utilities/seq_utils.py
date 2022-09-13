import numpy as np
import math
import kipoiseq

def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

def rev_comp_sequence(seq):
    comp = {"A":"T","T":"A","G":"C","C":"G","N":"N"}
    seq = seq[::-1]
    seq = "".join([comp[x] for x in seq])
    return seq

def rev_comp_one_hot(one_hot_seq):
      return np.flip(one_hot_seq,axis=(1,2))
    
def compute_offset_to_center_landmark(landmark,insert):
    mid = len(insert)//2
    return mid - landmark

def shuffle_string(string):
    string_as_list = list(string)
    np.random.shuffle(string_as_list)
    return "".join(string_as_list)

def extract_refseq_centred_at_landmark(landmark_interval, 
                                       fasta_extractor, 
                                       shift_five_end,
                                       SEQUENCE_LENGTH=15000,
                                       PADDING=10,
                                       binsize=128,
                                       rev_comp=False):
    assert len(landmark_interval) == 1
    five_end_len = math.ceil((SEQUENCE_LENGTH)/2) + shift_five_end
    three_end_len = math.floor((SEQUENCE_LENGTH)/2) - shift_five_end
    interval =  kipoiseq.Interval(chrom=landmark_interval.chrom,
                                 start=landmark_interval.start - five_end_len,
                                 end=landmark_interval.start + three_end_len)
    seq = fasta_extractor.extract(interval)
    assert (_seq_len := len(seq)) == SEQUENCE_LENGTH, \
        f"sequence should be {SEQUENCE_LENGTH} long, has length {_seq_len}"
    # check that central nucleotide is indeed in the center
    landmark_seq = fasta_extractor.extract(landmark_interval)
    assert landmark_seq == seq[math.ceil((SEQUENCE_LENGTH)/2) + shift_five_end]
    # compute the bins intersecting the insert sequence
    if not rev_comp:
        minbin = (five_end_len-PADDING)//binsize
        maxbin = ((five_end_len-PADDING)+1)//binsize
        landmarkbin = ((five_end_len-PADDING))//binsize
    else:
        seq = rev_comp_sequence(seq)
        minbin = (three_end_len-PADDING)//binsize
        maxbin = ((three_end_len-PADDING)+1)//binsize
        landmarkbin = ((three_end_len-PADDING))//binsize
    return seq, minbin, maxbin, landmarkbin

def insert_variant_centred_on_tss(tss_interval,
                                  variant,
                                  allele,
                                  fasta_extractor, 
                                  shift_five_end,
                                  SEQUENCE_LENGTH=15000,
                                  PADDING=10,
                                  binsize=128,
                                  rev_comp=False):
    assert len(tss_interval) == 1
    assert (tss_interval.chrom == variant.chrom)
    if allele == "ref":
        # we can just extract a window around the TSS position
        return extract_refseq_centred_at_landmark(tss_interval, 
                                                  fasta_extractor, 
                                                  shift_five_end,
                                                  SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                                  PADDING=PADDING,
                                                  binsize=128,
                                                  rev_comp=rev_comp)
    tss_position = tss_interval.start
    var_seq = variant.alt
    variant_ref_five_end = variant.start # 0-based
    variant_ref_three_end = variant.start+len(variant.ref) # 0-based
    if tss_position >= variant_ref_three_end:
        # tss is to the right of variant: 5' ==|VAR|==TSS==== 3'
        # part from the position right after the variant to the end of the sequence won't change
        # as TSS should be centred we need sequence from var three end to tss and then half of Enformer window
        # i.e. here we construct |==TSS==== 3'
        tss_distance = tss_position - variant_ref_three_end
        three_end_len = tss_distance + math.floor(SEQUENCE_LENGTH/2) - shift_five_end
        three_end_interval = kipoiseq.Interval(
            chrom=variant.chrom,
            start=variant_ref_three_end,
            end=variant_ref_three_end+three_end_len
        )
        # for the five end, we first construct the sequence up to the variant
        # for this we need to know the actual length of the allele we are about to insert
        # we need seq_len/2 - (distance var to tss) - len(allele to be inserted)
        # so here we construct ==|VAR
        five_end_len = math.ceil(SEQUENCE_LENGTH/2) - tss_distance - len(var_seq) + shift_five_end
        five_end_interval = kipoiseq.Interval(
            chrom=variant.chrom,
            start=variant_ref_five_end - five_end_len,
            end=variant_ref_five_end
        )
    elif tss_position < variant_ref_five_end:
        # tss is to the left of variant 5' ====TSS==|VAR|== 3'
        # this is just the above reversed
        # one can probably generalize this to treat both cases symetrically
        # construct first 5' ====TSS==|
        tss_distance = variant_ref_five_end - tss_position
        five_end_len = tss_distance + math.ceil(SEQUENCE_LENGTH/2) + shift_five_end
        five_end_interval = kipoiseq.Interval(
            chrom=variant.chrom,
            start=variant_ref_five_end - five_end_len,
            end=variant_ref_five_end
        )
        # next we construct VAR|== 3'
        three_end_len = math.floor(SEQUENCE_LENGTH/2) - tss_distance - len(var_seq) - shift_five_end
        three_end_interval = kipoiseq.Interval(
            chrom=variant.chrom,
            start=variant_ref_three_end,
            end=variant_ref_three_end + three_end_len
        )
    else:
        # there is a special case that the variant touches the tss
        # we exclude this case by design 
        # simply because it is not clear what it even means
        assert False,  f"tss_interval: {tss_interval} / variant: {variant}"# this should never occur
    five_end_seq = fasta_extractor.extract(five_end_interval)
    three_end_seq = fasta_extractor.extract(three_end_interval)
    # assemble flanks and insert variant allele into final sequence
    modified_sequence = five_end_seq + var_seq + three_end_seq
    # aggressive asserting
    assert (_mod_len := len(modified_sequence)) == SEQUENCE_LENGTH, \
        f"modified_sequence should be {SEQUENCE_LENGTH} long, has length {_mod_len}"
    # make sure the tss stayed in the middle
    tss_seq = fasta_extractor.extract(tss_interval)
    tss_local_pos = math.ceil((SEQUENCE_LENGTH)/2) + shift_five_end
    assert tss_seq == modified_sequence[tss_local_pos]
    # make sure the variant is where it is supposed to be
    if tss_position > variant_ref_three_end:
        assert var_seq == modified_sequence[tss_local_pos-tss_distance-len(var_seq):tss_local_pos-tss_distance]
    else:
        assert var_seq == modified_sequence[tss_local_pos+tss_distance:tss_local_pos+tss_distance+len(var_seq)]
    # compute the bins intersecting the TSS (unsure here)
    minbin = (math.ceil(SEQUENCE_LENGTH/2) - PADDING + shift_five_end - 128)//binsize
    maxbin = (math.ceil(SEQUENCE_LENGTH/2) - PADDING + shift_five_end + 128)//binsize
    landmarkbin = (math.ceil(SEQUENCE_LENGTH/2) - PADDING + shift_five_end)//binsize
    if rev_comp:
        modified_sequence = rev_comp_sequence(modified_sequence)
        landmarkbin = (SEQUENCE_LENGTH - 2*PADDING)//128 - landmarkbin - 1
        minbin = (SEQUENCE_LENGTH - 2*PADDING)//128 - minbin - 1
        maxbin = (SEQUENCE_LENGTH - 2*PADDING)//128 - maxbin - 1
    return modified_sequence, minbin, maxbin, landmarkbin

def pad_sequence(insert,
                shift_five_end=0,
                SEQUENCE_LENGTH=15000,
                PADDING=10,
                binsize=128,
                landmark=0,
                rev_comp=False):
    # compute required sequence
    five_end_len = math.ceil((SEQUENCE_LENGTH - len(insert))/2) + shift_five_end
    three_end_len = math.floor((SEQUENCE_LENGTH - len(insert))/2) - shift_five_end
    # assemble
    five_end_seq = "N"*five_end_len
    three_end_seq = "N"*three_end_len
    # assemble
    modified_sequence = five_end_seq + insert + three_end_seq
    assert len(modified_sequence) == SEQUENCE_LENGTH
    # compute the bins intersecting the insert sequence
    if not rev_comp:
        minbin = (five_end_len-PADDING)//binsize
        maxbin = ((five_end_len-PADDING)+len(insert))//binsize
        landmarkbin = ((five_end_len-PADDING) + landmark)//binsize
    else:
        modified_sequence = rev_comp_sequence(modified_sequence)
        assert modified_sequence[three_end_len + len(insert) - landmark - 1] == rev_comp_sequence(insert[landmark])
        landmark = len(insert) - landmark - 1
        minbin = (three_end_len-PADDING)//binsize
        maxbin = ((three_end_len-PADDING)+len(insert))//binsize
        landmarkbin = ((three_end_len-PADDING) + landmark)//binsize
    return modified_sequence, minbin, maxbin, landmarkbin

def insert_sequence_at_landing_pad(insert,
                                   lp_interval,
                                   fasta_extractor,
                                   mode="center",
                                   shift_five_end=0,
                                   SEQUENCE_LENGTH=1000,
                                   PADDING=1000,
                                   binsize=128,
                                   landmark=0,
                                   rev_comp=False,
                                   shuffle=False):
    """Inserts given insert into a sequence extracted from a fasta file

    Arguments:
    insert -- sequence to be inserted
    lp_interval -- landing pad interval, coordinates for the insertion within the genome
    fasta_extractor -- the fasta extractor object used to extract the genomic region
    mode -- whether to insert between landing pad ("center") or replace it ("replace")
    shift_five_end -- amount by which the insert will be shifted
    SEQUENCE_LENGTH -- sequence length for the final output
    PADDING -- starting position of the first enformer bin
    binsize -- size of an enformer bin
    landmark -- "point of interest" in the insert (e.g. a variant)
    rev_comp -- return data for reverse complemented version of sequence

    Returns:
    modified_sequence -- the constructed genomic region with the insert inserted
    minbin -- the first enformer bin which intersects with the insert
    maxbin -- the last enformer bin which intersects with the insert
    landmarkbin -- the enformer bin which intersects with the landmark
    """
    # calculate length of the flanks to be extracted in the next step s.t. insert fits
    # into final sequence and shift is taken into account
    five_end_len = math.ceil((SEQUENCE_LENGTH - len(insert))/2) + shift_five_end
    three_end_len = math.floor((SEQUENCE_LENGTH - len(insert))/2) - shift_five_end
    # extract the genomic sequences for both flanks
    if mode == "center": # integrate the insert sequence at landing pad midpoint
        # cut lp_interval in half s.t. insert can be inserted in between genomic sequence
        lp_midpoint = (lp_interval.start + lp_interval.stop)//2
        five_end_seq_interval = kipoiseq.Interval(chrom=lp_interval.chrom,
                                                  start=lp_midpoint-five_end_len,
                                                  end=lp_midpoint)
        three_end_seq_interval = kipoiseq.Interval(chrom=lp_interval.chrom,
                                                   start=lp_midpoint,
                                                   end=lp_midpoint+three_end_len)
    else: # replace the entire landing pad sequence with the insert
        # cut out the landing pad from genomic sequence, insert the insert instead
        five_end_seq_interval = kipoiseq.Interval(chrom=lp_interval.chrom,
                                                  start=lp_interval.start-five_end_len,
                                                  end=lp_interval.start)
        three_end_seq_interval = kipoiseq.Interval(chrom=lp_interval.chrom,
                                                   start=lp_interval.end,
                                                   end=lp_interval.end+three_end_len)
    five_end_seq = fasta_extractor.extract(five_end_seq_interval)
    three_end_seq = fasta_extractor.extract(three_end_seq_interval)
    if shuffle:
        five_end_seq = shuffle_string(five_end_seq)
        three_end_seq = shuffle_string(three_end_seq)
    # assemble flanks and insert into final sequence
    modified_sequence = five_end_seq + insert + three_end_seq
    assert (_mod_len := len(modified_sequence)) == SEQUENCE_LENGTH, \
        f"modified_sequence should be {SEQUENCE_LENGTH} long, has length {_mod_len}"
    # compute the bins intersecting the insert sequence
    if not rev_comp:
        minbin = (five_end_len-PADDING)//binsize
        maxbin = ((five_end_len-PADDING)+len(insert))//binsize
        landmarkbin = ((five_end_len-PADDING) + landmark)//binsize
    else:
        modified_sequence = rev_comp_sequence(modified_sequence)
        assert modified_sequence[three_end_len + len(insert) - landmark - 1] == rev_comp_sequence(insert[landmark])
        landmark = len(insert) - landmark - 1
        minbin = (three_end_len-PADDING)//binsize
        maxbin = ((three_end_len-PADDING)+len(insert))//binsize
        landmarkbin = ((three_end_len-PADDING) + landmark)//binsize
    return modified_sequence, minbin, maxbin, landmarkbin

def create_variant_sequence(variant, interval, var_extractor):
    alt_seq = var_extractor.extract(interval, [variant], anchor=interval.center() - interval.start)
    return alt_seq
