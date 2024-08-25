import os
import numpy as np
import sys as sys
import matplotlib.pyplot as plt

#updated 2022_04_15_to recognize ACE, NME and HSD as amino acid names
#updated 2022_06_03 to use CA distances for .lcc measurements (rather than geometric centers of amino acids)

class Trajectory (object):
   
    """This package reads Amber trajectory files and extracts various types of 
    information from it, including: 
        * sequence
        * length of sequence
        * total number of frames
        * acidic residue positions
        * basic residue positions
        * local compaction curves (lcc)"""
    
    def __init__ (self, name, prmtop, traj, start):
        self.name = name
        self.prmtop = prmtop
        self.traj = traj
        self.start = start
        
        print (f'Name of experiment:\t\t {self.name}')
        print (f'Name of .prmtop file:\t\t {self.prmtop}')
        print (f'Name of .nc file:\t\t {self.name}')
        print (f'Sequence start position:\t {self.start}')
        
        frames = 'cpptraj -p ' + self.prmtop + ' -y ' + self.traj + ' -tl'      
        os.system(frames) # run cpptraj in bash to get number of frames in trajectory file
        print('')

        f1 = open('temporary.cpptraj', 'w') # compose cpptraj script to get sequence from .prmtop file
        f1.write('parm ')
        f1.write(self.prmtop)
        f1.write('\n')
        f1.write('resinfo * out resinfo.txt\n')
        f1.close()
        os.system('cpptraj < temporary.cpptraj >/dev/null 2>&1') # run the script in bash, suppress console output
        self.sequence_read = np.loadtxt('resinfo.txt',dtype = 'str') # read in result
        os.system ('rm temporary.cpptraj resinfo.txt') # delete temporary files
        self.sequence = self.sequence_read [:,1]
        self.sequence = np.asarray(self.sequence)
        
        # dictionary for converting three amino acid letter code to single letter code
        d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 
             'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 'HID': 'H', 'HIE': 'H',
             'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 
             'MET': 'M', 'ACE': 'n', 'NME': 'c', 'HSD': 'H'}
        self.single_letter = ''
        for i in range (0,len(self.sequence)):
            self.single_letter += d[self.sequence[i]]
        self.length = len(self.single_letter)
        print(self.single_letter)
        self.absolute_position = [self.start, self.start + self.length - 1]
        print("Position =", self.absolute_position[0], "-", self.absolute_position[1], 
              "(", self.length, " aa long )", "\n\n")
    
    def get_sequence (self):
        return self.single_letter, self.length, self.absolute_position
    
    def get_all_acidics(self):
        asp_positions = []
        glu_positions = []
        self.all_acidics = [] 
        for s,t in enumerate(self.single_letter):
            if t == 'D':
                asp_positions.append (s)
            elif t == 'E':
                glu_positions.append (s)
        self.acidic_1 = np.asarray (asp_positions) + self.start
        self.acidic_2 = np.asarray (glu_positions) + self.start        
        # combine and sort all acidics
        self.all_acidics = np.hstack((self.acidic_1, self.acidic_2))
        self.all_acidics = np.sort(self.all_acidics) # sort in ascending order
        self.all_acidics = self.all_acidics.astype(int)       
        return self.all_acidics
    
    def get_all_basics(self):
        arg_positions = []
        his_positions = []
        lys_positions = []
        self.all_basics = []     
        for s,t in enumerate(self.single_letter):
            if   t == 'R':
                arg_positions.append (s)
            elif t == 'H':
                his_positions.append (s)
            elif t == 'K':
                lys_positions.append (s)
        self.basic_1  = np.asarray (arg_positions) + self.start
        self.basic_2  = np.asarray (his_positions) + self.start
        self.basic_3  = np.asarray (lys_positions) + self.start
        # combine and sort all basics
        self.all_basics = np.hstack((self.basic_1, self.basic_2))
        self.all_basics = np.hstack((self.all_basics, self.basic_3))
        self.all_basics = np.sort(self.all_basics) # sort in ascending order
        self.all_basics = self.all_basics.astype(int)
        return self.all_basics
    
    def get_local_distances (self, window): # read window size
        # write cpptraj script to file
        self.window = window
        f2 = open('distance.cpptraj', 'w')
        f2.write('parm ')
        f2.write(self.prmtop)
        f2.write('\n')
        f2.write('trajin ')
        f2.write(self.traj)
        f2.write('\n')
        upper_limit = self.length + 1 - window #max protein length + 1
        for x in range (1, upper_limit + 1):
            f2.write('distance :')
            f2.write(str(x) + '@CA') #use CA as reference atom in each residue
            f2.write(' :')
            #if x > protein_length - window:
            #    position = (x - protein_length + window)
            #    f.write (str(position))
            #else:
            f2.write(str(x + self.window) + '@CA') #use CA as reference atom in each residue
            f2.write(' out ')
            f2.write(self.name)
            f2.write('_window_')
            variable = str(self.window).zfill(3)
            f2.write(variable)
            f2.write('.lcc\n')
        f2.close()
        os.system ('cpptraj < distance.cpptraj >/dev/null 2>&1') # supress console output
        os.system ('rm distance.cpptraj')
        # remove first column of cpptraj-generated file containing frame numbers
        name = self.name + '_' + 'window' + '_' + variable + '.lcc'
        data = np.loadtxt(name)
        new_data = data [:,1:]
        print ('Shape of .lcc data set generated:', new_data.shape, '\n')
        np.savetxt (name, new_data)
        
def create_local_compaction_plot (lcc_data, start, stride, alpha, color): # read in lcc data
    if lcc_data.find('lcc') == -1:
        print('This is not a .lcc file!')
        pass
    else:
        position = lcc_data.find('lcc')
        window_distance = int(lcc_data[position - 4 : position -1]) # due to zero padding the number can be recovered by position
        name = lcc_data[0 : position - 12] 
        print ('Window size identified from file name:', window_distance)
        print ('Start position:   ', start)
        print ('Trajectory Stride:', stride)
        print ('Alpha:            ', alpha)
        print ('Color:            ', color)
        data = np.loadtxt(lcc_data)
        print (data.shape[0], "rows with", data.shape[1], "data points/row") # column 0 of row is row number -> ignore
            
        # plot out data
        x_axis = np.arange(0, data.shape[1]) + start + int (window_distance/2)# adjusted for start position
        for x in range(0, data.shape[0], stride): #int(self.stride)):
            plt.plot (x_axis, data [x, :], alpha = alpha, color = color);

        plt.xlabel('Amino Acid Sequence Position')  
        plt.ylabel('Distance ($\AA$)')  #Angstrom
        plt.title('Local Compaction Plot - ' + str(window_distance) + ' aa Window')
        figure_name = name + "_window_" + str(window_distance) + "_s" + str(stride) + "_a" + str(alpha) + "_lcp.png"
        plt.savefig(figure_name, dpi = 300)
        plt.show()
        print("Residue positions shown:", x_axis)
            
def quantitate_complexity (file_name):
    '''Takes in a lcp position, returns the compaction rate reflecting Kolmogorov complexity'''
    
    # Kolmogorov complexity/randomness; descriptive complexity; algorithmic complexity
    # Kolmogorov, AN (1965) Three approaches to the quantitative definition of information. Problemy Peredachi Informatsii 1: 3–11
    # For mathematically non-trivial reasons, Kolmogorov complexity is actually incomputable (Kolmogorov, 1965).
    # Kolmogorov complexity can, however, be approximated using file compression programs
    # Modern file compression programs such as gzip use a variant of adaptive entropy estimation that approximates Kolmogorov complexity. In fact, the Kolmogorov complexity of a given text file is the file size of the compressed version of this file (Li et al., 2004; Ziv and Lempel, 1977).
    # Text compression algorithms such as gzip compress text strings by describing new strings on the basis of previously encountered (sub)strings, and so measure the amount of information and redundancy in a given text string (Juola, 2008: 93).

    # see https://journals.sagepub.com/doi/full/10.1177/0267658316669559
    # http://www.gzip.org/
    original_size = []
    gzip_compressed_size = []

    data = np.loadtxt (file_name)
    print ('File name read in:', file_name, '\n')
    for x in range (0, 50):
        name = str(x)
        np.savetxt(name, data[:,x])
        original = os.stat(name).st_size # obtain size of original file
        original_size.append (original)    
        command_1 = 'gzip ' + name
        os.system(command_1) # use gzip
        gz_name = name + ".gz"
        compressed = os.stat(gz_name).st_size # determine size after gzip compression
        gzip_compressed_size.append(compressed)
        command_3 = 'rm ' + gz_name
        os.system(command_3)
    # convert arrays to numpy arrays
    original_size = np.array(original_size)
    gzip_compressed_size = np.array (gzip_compressed_size)
    print ('Number of positions analyzed before (', len(original_size),') and after complexity analysis (', len(gzip_compressed_size),')\n')
    ratio = gzip_compressed_size/original_size
    median = np.median(ratio)
    stdev = np.std(ratio)
    print ('The median of the compressed data set is', round(median,2))
    print ('The standard deviation of the compressed data set is', round(stdev,2))
    x_range = np.arange(0, len(ratio))
    y_range_median = np.ones(50) * median
    y_range_stedev = np.ones(50) * stdev
    plt.xlabel ('Position')
    plt.ylabel ('Compression Ratio')
    plt.ylim(0.2, 0.24)
    plt.plot(x_range, y_range_median, ".", linewidth = 0.1, color = "coral");
    plt.plot(x_range, y_range_median + y_range_stedev, linewidth = 1, color = "gainsboro");
    plt.plot(x_range, y_range_median - y_range_stedev, linewidth = 1, color = "gainsboro");
    plt.plot (ratio, color = 'navy');
    new_file_name =  file_name + '.compacted.png'
    plt.savefig (new_file_name, dpi = 300)
    return (original_size, gzip_compressed_size, median)
