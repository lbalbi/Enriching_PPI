import pandas, torch, os, esm
from transformers import T5Tokenizer, T5EncoderModel
from Bio.PDB import PDBParser, MMCIFParser
import xpdb


def parse_pdb(pdb_file):

    if pdb_file.endswith(".pdb"): sloppyparser = PDBParser(PERMISSIVE=True, structure_builder=xpdb.SloppyStructureBuilder())
    elif pdb_file.endswith(".cif"): sloppyparser = MMCIFParser(structure_builder=xpdb.SloppyStructureBuilder())
    structure = sloppyparser.get_structure("MD_system", pdb_file)
    sloppyio = xpdb.SloppyPDBIO()
    sloppyio.set_structure(structure)
    structure_ = []
    for atom in structure.get_atoms():
        structure_.append(str(atom).strip("<Atom ").strip(">").lower())
    return structure_


def direct_LM_structures(dir = "/mnt/data2/lbalbi/huri_pdbs/"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

    files = os.listdir(dir)
    paths = [path for path in files if path.endswith(".pdb") or path.endswith(".cif")]
    sequence_examples, embeddings, protein_embedding, count = [],[], dict(), 0

    for path in paths:
        if protein_embedding.get(path.split("-")[1]) == None: protein_embedding[path.split("-")[1]] = []
        else: continue
        torch.cuda.empty_cache()
        sequence_examples = parse_pdb(dir + path)
        sequence_examples = [" ".join(sequence) for sequence in sequence_examples] #  introduce white-space between all sequences (AAs and 3Di)
        sequence_examples = [ "<fold2AA>" + " " + s for s in sequence_examples ] #  to embed 3Di you need to prepend "<fold2AA>"
        ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest",return_tensors='pt')
        with torch.no_grad():
            embedding_repr = model(ids.input_ids.to(device), attention_mask=ids.attention_mask.to(device))
        torch.cuda.empty_cache()
            # try: embedding_repr = model(ids.input_ids.to(device), attention_mask=ids.attention_mask.to(device))
            # except torch.cuda.OutOfMemoryError: 
            #     model = model.to("cpu")
            #     embedding_repr = model(ids.input_ids, attention_mask=ids.attention_mask)
            #     torch.cuda.empty_cache()
            #     model = model.to(device)

        embedding_repr = embedding_repr.last_hidden_state[-1]
        torch.cuda.empty_cache()
        emb_0_per_protein = embedding_repr.mean(dim=0)
        count += 1
        if count %30 == 0: print(count)
        protein_embedding[path.split("-")[1]].append(emb_0_per_protein)
    return protein_embedding


def protein_structures(proteins, mappings_, dir = "mnt/data2/lbalbi/huri_pdbs/"):
    proteins_list, dict_proteins, mappings_dict, mappings = pandas.read_csv(proteins, index_col=False)["protein id"], dict(), dict(), []
    for prot in proteins_list: dict_proteins[prot] = [prot]
    for map in mappings_:  
        protein_mappings = pandas.read_csv(map, index_col=False, sep="\t")
        for k in range(len(protein_mappings["Entry"])): 
            if dict_proteins.get(protein_mappings["Entry"][k]) == None: dict_proteins[protein_mappings["Entry"][k]] = []
            dict_proteins[protein_mappings["Entry"][k]].append(protein_mappings["From"][k])
        
    list_files = os.listdir(dir)
    for file in list_files:
        if file.endswith(".pdb") or file.endswith(".cif"):
            if dict_proteins.get(file.split("-")[1]) != None:
                for prot in dict_proteins[file.split("-")[1]]:
                    if mappings_dict.get(prot) == None: mappings_dict[prot] = []
                    mappings_dict[prot].append(file)

    for prot in proteins_list:
        if mappings_dict.get(prot) != None: mappings.append(mappings_dict[prot])
        elif mappings_dict.get(prot) == None: print(prot)
    return mappings


def protein_embeddings(model_, dir = "/mnt/data2/lbalbi/huri_pdbs/"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if model_.endswith("esm_if1_gvp4_t16_142M_UR50"):

        count, protein_embed, files = 0, dict(), os.listdir(dir)
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        paths, chain_ids = [file for file in files if file.endswith(".cif") or file.endswith(".pdb")], ["A"] #["A","C","D","N","S","O","B","CG","H","L","E","V"]
        for path in paths:
            if protein_embed.get(path.split("-")[1]) == None: protein_embed[path.split("-")[1]] = []
            for chain in chain_ids:

                torch.cuda.empty_cache()
                structure_ = esm.inverse_folding.util.load_structure(dir + path, chain)
                coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure_)

                try:  rep = esm.inverse_folding.util.get_encoder_output(model.to(device), alphabet, coords)
                except torch.cuda.OutOfMemoryError: rep = esm.inverse_folding.util.get_encoder_output(model.to("cpu"), alphabet, coords)
                except: print(path.split("-")[1]); pass

                rep = rep.detach().cpu().mean(dim=0)
                print(rep.shape); exit()
                protein_embed[path.split("-")[1]].append(rep)

            count += 1
            if count % 100 == 0: print(count); 
            if protein_embed[path.split("-")[1]] == [] : print(path)
        del model, alphabet, structure_, coords
        return protein_embed


def protein_embeddings_v2(model, dir = "/mnt/data2/lbalbi/huri_pdbs/"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    import biolib
    foldseek = biolib.load(model)
    paths = [file for file in os.listdir(dir) if file.endswith(".cif") or file.endswith(".pdb")]
    for path in paths:
        foldseek_results = foldseek.cli(args='easy-search {} /data/PDB/pdb output.aln tmp --format-mode 4'.format(dir + path))
    print(foldseek_results); exit()



def process_embeddings(proteins, embeddings, mappings):
    protein_df, embedding_array = pandas.read_csv(proteins, index_col=False, sep=","), []
    
    for k in range(len(protein_df["protein id"])):
        mapped = False
        for j in mappings[k]: # mappings has list of files for each protein
            if embeddings.get(j.split("-")[1]) != None:
                if len(embeddings[j.split("-")[1]]) > 1: embedding_array.append(torch.mean(torch.stack(embeddings[j.split("-")[1]]),0).to("cpu"));break
                elif len(embeddings[j.split("-")[1]]) == 1: embedding_array.append(torch.mean(embeddings[j.split("-")[1]][0],0).to("cpu"));break
                else: print(j.split("-")[1], embeddings[j.split("-")[1]] , len(embeddings[j.split("-")[1]])); pass
    print(len(embedding_array))
    if len(embedding_array) == len(protein_df["protein id"]): return torch.stack(embedding_array, dim=0)


def clean_embeddings_(embed, model):
    
    for key in embed.keys():
        keep = []
        for content in key:
            if len(content) > 20: print(content)
            else: keep.append(content)
        embed[key] = list(set(keep))
    torch.save(embed, "embeddings_{}___.pt".format( model.split("/")[1]))



def main(proteins, model, mappings):
    if model == "facebook/esm_if1_gvp4_t16_142M_UR50":
        # structures = protein_structures(proteins, mappings)
        # torch.save(structures, "structures_{}___.pt".format(model.split("/")[1]))
        embeddings = protein_embeddings(model)
        # torch.save(embeddings, "embeddings_{}___.pt".format(model.split("/")[1]))
        # embeddings_ = process_embeddings(proteins, torch.load("embeddings_{}.pt".format(model.split("/")[1])), torch.load("structures_{}___.pt".format(model.split("/")[1])))
        # torch.save(embeddings_, "embeddings{}_{}___.pt".format(mappings[0].strip("mapping").strip(".tsv"), model.split("/")[1]))
    
    elif model == "Rostlab/ProstT5":
        embeddings = direct_LM_structures()
        # torch.save(embeddings, "embeddings_{}___.pt".format( model.split("/")[1]))
        # embeddings_ = process_embeddings(proteins, torch.load("/mnt/data2/lbalbi/embeddings_{}___.pt".format(model.split("/")[1])), torch.load("structures_{}___.pt".format(model.split("/")[1])))
        # torch.save(embeddings_, "embeddings{}_{}___.pt".format(mappings[0].strip("mapping").strip(".tsv"), model.split("/")[1]))        

    elif model.startswith("Protein_Tools/"):
        embeddings = protein_embeddings_v2(model)

## LM model
model_esmif1 = "facebook/esm_if1_gvp4_t16_142M_UR50"
model_prost = "Rostlab/ProstT5"

model_foldseek = "Protein_Tools/foldseek"
model_proteinMPNN = "Protein_Tools/proteinMPNN"
model_alphafold = "Protein_Tools/af_design"

## dataset entities
ogb, ogb_match = "nodeidx2proteinid_ogb.csv", ".tsv"
huri, huri_match, huri_match2 = "entities.csv", "mapping_huri.tsv", "backup_sequences.tsv"


main(ogb, model_foldseek, [ogb_match])

##
# Foldseek - https://biolib.com/protein-tools/foldseek/
# ProteinMPNN - https://biolib.com/protein-tools/mpnn-scoring/     https://biolib.com/protein-tools/proteinMPNN/
# AlphaFold Design -  https://biolib.com/protein-tools/af-design/
# OmegaFold (sequence LLM) - https://biolib.com/protein-tools/omegafold/