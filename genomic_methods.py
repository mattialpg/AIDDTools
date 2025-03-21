from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import SeqIO, Align
from Bio.Align import substitution_matrices
aligner = Align.PairwiseAligner()
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

from collections import defaultdict

import pandas as pd
from Bio.Blast import NCBIXML
def blast_xml_to_df(xml_file):
    with open(xml_file, 'r') as file:
        blast_records = NCBIXML.parse(file)
        
        data = []
        for record in blast_records:
            query_id = record.query  # Query sequence ID
            query_length = record.query_length  # Query sequence length
        
            for alignment in record.alignments:
                subject_id = alignment.hit_id
                subject_def = alignment.hit_def
                subject_accession = alignment.accession
                subject_length = alignment.length
            
                for hsp in alignment.hsps[:1]:  #<-- Only the first HSP is considered
                    data.append({
                    "Query_ID": query_id,
                    "Subject_ID": subject_id,
                    "Subject_Definition": subject_def,
                    "Subject_Accession": subject_accession,
                    "Query_Length": query_length,
                    "Subject_Length": subject_length,
                    "Alignment_Length": hsp.align_length,
                    "Query_Location": f"{hsp.query_start}..{hsp.query_end}",
                    "Subject_Location": f"{hsp.sbjct_start}..{hsp.sbjct_end}",
                    "Query_Strand": '+' if hsp.query_start < hsp.query_end else '-',
                    "Subject_Strand": '+' if hsp.sbjct_start < hsp.sbjct_end else '-',
                    "E-value": hsp.expect,
                    "Identity_%": round((hsp.identities/hsp.align_length)*100, 1) if hsp.align_length > 0 else 0,
                    # "Coverage": round((hsp.align_length / hsp.align_length * 100), 2),
                    "Gaps": hsp.gaps,
                    "Positive_Matches": hsp.positives,
                    "Query_Sequence": hsp.query,
                    "Match_String": hsp.match,
                    "Subject_Sequence": hsp.sbjct,})
    df = pd.DataFrame(data)
    return df

def remove_mismatches(alignment):
    segments = []
    count = 0
    current_type = None

    alignment = str(alignment).strip().split("\n")
    
    for char in alignment[1]:
        if char == '-':  # Count as gap
            if current_type == 'gap':
                count += 1
            else:
                if current_type:  # Append the previous residue/gap count
                    segments.append(f"{count}{current_type}")
                current_type = 'gap'
                count = 1
        elif char == '|' or char == '.':  # Count as residue
            if current_type == 'res':
                count += 1
            else:
                if current_type:  # Append the previous residue/gap count
                    segments.append(f"{count}{current_type}")
                current_type = 'res'
                count = 1
        else:  # For actual residues (other than '.' or '|')
            if current_type == 'res':
                count += 1
            else:
                if current_type:  # Append the previous residue/gap count
                    segments.append(f"{count}{current_type}")
                current_type = 'res'
                count = 1
    if current_type:
        segments.append(f"{count}{current_type}")
    # print(segments)

    if len(segments) == 1:
        return alignment

    decision = []
    for i in range(len(segments)):
        ipso = int(segments[i][:-3])
        preceding = int(segments[i-1][:-3]) if i > 0 else 100000
        following = int(segments[i+1][:-3]) if i+1 < len(segments) else 100000

        if 'gap' in segments[i]:
            if i == 0 or i == len(segments)-1:
                decision.append('Delete')
                continue
            
            if preceding > 0.2 * ipso and following > 0.2 * ipso:
                decision.append('Keep')
            else:
                decision.append('Delete')
        else:
            if ipso > 0.2 * preceding or ipso > 0.2 * following:
                decision.append('Keep')
            else:
                decision.append('Delete')
    # print(decision)

    # Now apply the decisions to the alignment
    modified_alignment = []
    for line in alignment:
        modified_line = ''
        i = 0
        for seg, dec in zip(segments, decision):
            seq = line[i:i+int(seg[:-3])]
            if dec == 'Keep':
                modified_line += seq
            i += int(seg[:-3])
        modified_alignment.append(modified_line)
    
    return modified_alignment

def find_mutations(alignment):
    query_index = 0
    mutations = []

    alignment = remove_mismatches(alignment)

    for idx, (q_char, t_char) in enumerate(zip(alignment[0], alignment[2])):
        if q_char != '-':
            query_index += 1
        
        # Substitution
        if alignment[1][idx] == '.':
            mutations.append(f"{q_char}{query_index}{t_char}")
        
        # Deletion (gap in target sequence)
        elif t_char == '-':
            mutations.append(f"Δ{query_index}")
        
        # Insertion (gap in query sequence)
        elif q_char == '-':
            gap_start = query_index
            # Find the length of consecutive gaps in the query sequence
            while idx + 1 < len(alignment[0]) and alignment[0][idx + 1] == '-':
                idx += 1
            
            gap_end = query_index + (idx - gap_start)
            mutations.append(f"{alignment[0][gap_start-1]}{gap_start}_{alignment[0][gap_end+1]}{gap_end}ins{t_char}")

    # Reduce notation
    mutation_dict = defaultdict(list)
    deletion_dict = defaultdict(list)
    reduced_mutations = []
    
    # Step 1: Sort mutations into insertions and deletions
    for mutation in mutations:
        if 'ins' in mutation:  # Handle insertion mutations
            position, insertion = mutation.rsplit('ins', 1)
            mutation_dict[position].append(insertion)
        elif mutation.startswith('Δ'):  # Handle deletion mutations
            position = int(mutation[1:])  # Convert to integer to handle gaps
            deletion_dict[position].append(position)
        else:
            # Non-insertion and non-deletion mutations are added as they are
            reduced_mutations.append(mutation)
    
    # Step 2: Combine insertions for each position
    for position, insertions in mutation_dict.items():
        combined_insertion = ''.join(sorted(set(insertions)))  # Remove duplicates and combine
        reduced_mutations.append(f"{position}ins{combined_insertion}")
    
    # Step 3: Combine consecutive deletions into a range (e.g., Δ17-18)
    sorted_deletions = sorted(deletion_dict.keys())
    if sorted_deletions:
        # Identify consecutive deletions
        consecutive_deletions = []
        start = sorted_deletions[0]
        end = start

        for i in range(1, len(sorted_deletions)):
            if sorted_deletions[i] == end + 1:  # Check if consecutive
                end = sorted_deletions[i]
            else:
                # Non-consecutive deletion, add the previous range
                if start == end:
                    consecutive_deletions.append(f"Δ{start}")
                else:
                    consecutive_deletions.append(f"Δ{start}-{end}")
                start = sorted_deletions[i]
                end = start

        # Add the last segment
        if start == end:
            consecutive_deletions.append(f"Δ{start}")
        else:
            consecutive_deletions.append(f"Δ{start}-{end}")
        
        # Add all concatenated deletions
        reduced_mutations.extend(consecutive_deletions)

    return reduced_mutations

# def calc_percent_similarity(alignment):   
#     seq1 = alignment.query
#     seq2 = alignment.target
    
#     shorter_seq = min(len(seq1), len(seq2))
#     max_possible_score = sum(aligner.substitution_matrix[(res, res)]
#                              for res in seq1[:shorter_seq])
    
#     percent_similarity = (alignment.score / max_possible_score) * 100
#     return round(percent_similarity, 1)

def calc_identity(alignment):
    alignment_list = str(alignment).strip().split("\n")
    matches = alignment_list[1].count('|')
    alignment_length = len(alignment_list[1].replace("-", ""))
    identity = matches / alignment_length * 100
    return round(identity, 1)

def fix_prokka_gbk(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(f"{folder}/{contig}.gbk", 'w') as f:
        for line in lines:
            if line.startswith("LOCUS"):
                for id in IDs:
                    if id in line and not line[line.index(id) + len(id):].startswith(" "):
                        line = line.replace(id, f"{id} ")
            f.write(line)