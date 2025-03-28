import os
import re
import csv
import math
import torch
import argparse

import numpy as np
from tqdm import tqdm


def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)


def match_feature(x, all_for_assign):
    x_p = np.zeros((len(x), 7))
    
    for j in range(len(x)):
        if x[j] == 'ALA':
            x_p[j] = all_for_assign[0, :]
        elif x[j] == 'CYS':
            x_p[j] = all_for_assign[1, :]
        elif x[j] == 'ASP':
            x_p[j] = all_for_assign[2, :]
        elif x[j] == 'GLU':
            x_p[j] = all_for_assign[3, :]
        elif x[j] == 'PHE':
            x_p[j] = all_for_assign[4, :]
        elif x[j] == 'GLY':
            x_p[j] = all_for_assign[5, :]
        elif x[j] == 'HIS':
            x_p[j] = all_for_assign[6, :]
        elif x[j] == 'ILE':
            x_p[j] = all_for_assign[7, :]
        elif x[j] == 'LYS':
            x_p[j] = all_for_assign[8, :]
        elif x[j] == 'LEU':
            x_p[j] = all_for_assign[9, :]
        elif x[j] == 'MET':
            x_p[j] = all_for_assign[10, :]
        elif x[j] == 'ASN':
            x_p[j] = all_for_assign[11, :]
        elif x[j] == 'PRO':
            x_p[j] = all_for_assign[12, :]
        elif x[j] == 'GLN':
            x_p[j] = all_for_assign[13, :]
        elif x[j] == 'ARG':
            x_p[j] = all_for_assign[14, :]
        elif x[j] == 'SER':
            x_p[j] = all_for_assign[15, :]
        elif x[j] == 'THR':
            x_p[j] = all_for_assign[16, :]
        elif x[j] == 'VAL':
            x_p[j] = all_for_assign[17, :]
        elif x[j] == 'TRP':
            x_p[j] = all_for_assign[18, :]
        elif x[j] == 'TYR':
            x_p[j] = all_for_assign[19, :]
            
    return x_p


def read_atoms(file, chain="."):
    pattern = re.compile(chain)

    atoms = []
    ajs = []
    
    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ajs_id = line[17:20]
                atoms.append((x, y, z))
                ajs.append(ajs_id)
                
    return atoms, ajs

            
def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms)-2):
        for j in range(i+2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i, j))
                contacts.append((j, i))
    return contacts


def knn(atoms, k=5):
    
    x = np.zeros((len(atoms), len(atoms)))
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            x[i, j] = dist(atoms[i], atoms[j])
    index = np.argsort(x, axis=-1)
    
    contacts = []
    for i in range(len(atoms)):
        num = 0
        for j in range(len(atoms)):
            if index[i, j] != i and index[i, j] != i-1 and index[i, j] != i+1:
                contacts.append((i, index[i, j]))
                num += 1
            if num == k:
                break
            
    return contacts


def pdb_to_cm(file, threshold):
    atoms, x = read_atoms(file)
    r_contacts = compute_contacts(atoms, threshold)
    k_contacts = knn(atoms)
    return r_contacts, k_contacts, x


def data_processing(dataset):
    name_list = []
    files = os.listdir("STRING_AF2DB")
    for file in files:
        name_list.append(file)
        

    outFile = open('./processed_data/protein.{}.sequences.dictionary.csv'.format(dataset),'a+', newline='')
    writer = csv.writer(outFile, dialect='excel')

    num_protein = 0
    for line in tqdm(open('./protein.{}.sequences.dictionary.tsv'.format(dataset))):
        if line.split('\t')[0] + '.pdb' in name_list:
            writer.writerow([line.split('\t')[0], line.split('\t')[1]])
            num_protein += 1

    outFile.close()


    flag = 1
    file = open('./processed_data/protein.actions.{}.txt'.format(dataset),mode='w')
        
    for line in tqdm(open('./protein.actions.{}.txt'.format(dataset))):
        if flag == 1:
            file.write(line)
            flag = 0
        else:
            lines = line.strip().split('\t')
            if lines[0]+".pdb" in name_list and lines[1]+".pdb" in name_list:
                file.write(line)
        
    print("{} | #Protein: {}".format(dataset, num_protein))


    # Generate Adjacency Matrix
    distance = 10 # threshold
    prot_files = os.listdir("./STRING_AF2DB")
    all_for_assign = np.loadtxt("./all_assign.txt")

    node_list = []
    r_edge_list = []
    k_edge_list = []

    for lines in tqdm(open('./protein.{}.sequences.dictionary.tsv'.format(dataset))):
        
        line = lines.split('\t')[0]
        pdb_file_name = line + '.pdb'
        
        if pdb_file_name in prot_files:
            
            r_contacts, k_contacts, x = pdb_to_cm(open("./STRING_AF2DB/" + pdb_file_name, "r"), distance)
            x = match_feature(x, all_for_assign)
            
            node_list.append(x)
            r_edge_list.append(r_contacts)
            k_edge_list.append(k_contacts)

    np.save("./processed_data/protein.rball.edges.{}.npy".format(dataset), np.array(r_edge_list))
    np.save("./processed_data/protein.knn.edges.{}.npy".format(dataset), np.array(k_edge_list))
    torch.save(node_list, "./processed_data/protein.nodes.{}.pt".format(dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Implementation")
    parser.add_argument("--dataset", type=str, default="SHS27k")
    args = parser.parse_args()

    if not os.path.exists("./processed_data"):
        os.makedirs("./processed_data")

    data_processing(args.dataset)