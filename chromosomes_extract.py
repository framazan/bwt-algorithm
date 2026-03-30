#!/usr/bin/env python3
"""
Extract each chromosome from Athaliana_167_TAIR9.fa into separate FASTA files.
"""

import os
import sys

def extract_chromosomes(input_file):
    """
    Extract each chromosome from the input FASTA file into separate files.
    
    Args:
        input_file (str): Path to the input FASTA file
    """
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    current_file = None
    current_chr = None
    
    print(f"Processing {input_file}...")
    
    try:
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Check if this is a header line
                if line.startswith('>'):
                    # Close previous file if open
                    if current_file:
                        current_file.close()
                        print(f"Finished writing {current_chr}.fa")
                    
                    # Extract chromosome name from header
                    # Expected format: >Chr1 CHROMOSOME dumped from ADB: Feb/3/09 16:9; last updated: 2007-12-20
                    header_parts = line.split()
                    if len(header_parts) > 0:
                        chr_name = header_parts[0][1:]  # Remove the '>' character
                        current_chr = chr_name
                        
                        # Create new file for this chromosome
                        output_filename = f"{chr_name}.fa"
                        current_file = open(output_filename, 'w')
                        current_file.write(line + '\n')
                        print(f"Started writing {output_filename}")
                    else:
                        print(f"Warning: Malformed header at line {line_num}: {line}")
                        continue
                
                # This is a sequence line
                elif line and current_file:
                    current_file.write(line + '\n')
                
                # Progress indicator for large files
                if line_num % 100000 == 0:
                    print(f"Processed {line_num:,} lines...")
    
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)
    
    finally:
        # Close the last file if still open
        if current_file:
            current_file.close()
            print(f"Finished writing {current_chr}.fa")
    
    print("Chromosome extraction completed!")

def main():
    input_file = "Athaliana_167_TAIR9.fa"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    extract_chromosomes(input_file)

if __name__ == "__main__":
    main()
