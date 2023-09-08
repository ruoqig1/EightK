# basic_parameters
# Created by Antoine Didisheim, at 06.08.19
# job: store default basic_parameters used throughout the projects in single .py

import itertools
from enum import Enum
import numpy as np
import pandas as pd
import socket
import os



##################
# Enum
##################

class NewsSource(Enum):
    WSJ = 'wsj'

class OptModelType(Enum):
    OPT_125m ='facebook/opt-125M'
    OPT_350m ='facebook/opt-350M'
    OPT_1b3 ='facebook/opt-1.3b'
    OPT_2b6 ='facebook/opt-2.7b'
    OPT_6b7 ='facebook/opt-6.7b'
    OPT_13b ='facebook/opt-13b'
    OPT_175b ='facebook/opt-175b'



##################
# constant
##################


class Constant:
    if socket.gethostname() in ['HEC37827','3330L-214940-M']:
        # main_dir = '/media/antoine/ssd_ntfs//wsj_openai/'
        MAIN_DIR = './'
    elif socket.gethostname() in ['rdl-orlr7x.desktop.cloud.unimelb.edu.au']:
        MAIN_DIR = '/mnt/layline/project/eightk/'
    else:
        MAIN_DIR = '/data/gpfs/projects/punim2039/EightK/'
        HUGGING_DIR = '/data/gpfs/projects/punim2039/hugging/'

    FRED_API = 'cfc526395a5631650ec0b7ee96b149f4'

    DROP_RES_DIR = '/Users/adidisheim/Dropbox/Apps/Overleaf/EightKEarlyAnalysis/'

    LIST_ITEMS_TO_USE = [1.01, 1.02, 1.03, 1.04, 2.01, 2.03, 2.04, 2.05, 2.06, 3.02, 3.03, 4.01, 4.02, 5.01, 5.02, 5.03, 5.04, 5.05, 5.06, 5.07, 5.08, 6.01, 6.02, 6.03, 6.04, 6.05, 7.01, 8.01, 9.01]
    LIST_ITEMS_FULL = [1.01, 1.02, 1.03, 1.04, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 3.01, 3.02, 3.03, 4.01, 4.02, 5.01, 5.02, 5.03, 5.04, 5.05, 5.06, 5.07, 5.08, 6.01, 6.02, 6.03, 6.04, 6.05, 7.01, 8.01, 9.01]

    SECTIONS = {
        1.0: "Registrant's Business and Operations",
        2.0: "Financial Information",
        3.0: "Securities and Trading Markets",
        4.0: "Matters Related to Accountants and Financial Statements",
        5.0: "Corporate Governance and Management",
        6.0: "Asset-Backed Securities",
        7.0: "Regulation FD",
        8.0: "Other Events",
        9.0: "Financial Statements and Exhibits"
    }

    ITEMS = {
        1.01: "Entry into a Material Definitive Agreement",
        1.02: "Termination of a Material Definitive Agreement",
        1.03: "Bankruptcy or Receivership",
        1.04: "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
        2.01: "Completion of Acquisition or Disposition of Assets",
        2.02: "Results of Operations and Financial Condition",
        2.03: "Creation of a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement of a Registrant",
        2.04: "Triggering Events That Accelerate or Increase a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement",
        2.05: "Costs Associated with Exit or Disposal Activities",
        2.06: "Material Impairments",
        3.01: "Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; Transfer of Listing",
        3.02: "Unregistered Sales of Equity Securities",
        3.03: "Material Modification to Rights of Security Holders",
        4.01: "Changes in Registrant's Certifying Accountant",
        4.02: "Non-Reliance on Previously Issued Financial Statements or a Related Audit Report or Completed Interim Review",
        5.01: "Changes in Control of Registrant",
        5.02: "Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers; Compensatory Arrangements of Certain Officers",
        5.03: "Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year",
        5.04: "Temporary Suspension of Trading Under Registrant's Employee Benefit Plans",
        5.05: "Amendment to Registrant's Code of Ethics, or Waiver of a Provision of the Code of Ethics",
        5.06: "Change in Shell Company Status",
        5.07: "Submission of Matters to a Vote of Security Holders",
        5.08: "Shareholder Director Nominations",
        6.01: "ABS Informational and Computational Material",
        6.02: "Change of Servicer or Trustee",
        6.03: "Change in Credit Enhancement or Other External Support",
        6.04: "Failure to Make a Required Distribution",
        6.05: "Securities Act Updating Disclosure",
        7.01: "Regulation FD Disclosure",
        8.01: "Other Events (The registrant can use this Item to report events that are not specifically called for by Form 8-K, that the registrant considers to be of importance to security holders.)",
        9.01: "Financial Statements and Exhibits"
    }

    IS_VM = socket.gethostname()=='rdl-orlr7x.desktop.cloud.unimelb.edu.au'

##################
# params classes
##################

class DataParams:
    def __init__(self):
        self.base_data_dir = Constant.MAIN_DIR + 'data/'
        self.tar_wsj_dir = Constant.MAIN_DIR + 'data/wsj_tar/'
        self.raw_wsj_dir = Constant.MAIN_DIR + 'data/wsj_untar/'
        self.pickle_ind_df = Constant.MAIN_DIR + 'data/pickle_ind_df/'

        self.data_to_use_name = 'load_xy_ravenpack_monthly_vy'


class EncodingParams:
    def __init__(self):
        self.opt_model_type = OptModelType.OPT_125m
        self.news_source = NewsSource.WSJ

        #params related to running the encoding
        self.nb_chunks = 100
        self.save_chunks_size = 500
        self.chunk_to_run_id = 1

class RandomFeaturesParams:
    def __init__(self):
        self.max_rf = 50*1000
        self.gamma_list = [0.5,0.6,0.7,0.8,0.9,1.0]
        self.block_size_for_generation=1000
        self.start_seed=0
        self.voc_grid=[100, 200, 360, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
        self.para_nb_of_list_group=20
        self.para_id=0


class TrainerParams:
    def __init__(self):
        self.T_train = 360
        self.T_val = 36
        self.testing_window = 1
        self.shrinkage_list = np.linspace(1e-12,10,50)
        self.save_ins = False

        # this is the number of individual saving chunks.
        # by this we mean the number of individual df contianing some oos performance that will be saved before merged.
        # too big and we risk loosing some processing, too small and we will make a mess of the merging process.
        self.nb_chunks = 25
        #
        self.min_nb_chunks_in_cluster = 10

# store all basic_parameters into a single object
class Params:
    def __init__(self):
        self.name_detail = 'default'
        self.name = ''
        self.seed = 12345
        self.data = DataParams()
        self.enc = EncodingParams()
        self.train = TrainerParams()
        self.rf = RandomFeaturesParams()

    def get_encoding_save_name(self, temp_dir = False):
        if temp_dir:
            d = Constant.MAIN_DIR + f'enc_temp/{self.enc.news_source.name}/{self.enc.opt_model_type.name}/chunk_size{self.enc.save_chunks_size}/'
        else:
            d = Constant.MAIN_DIR + f'enc/{self.enc.news_source.name}/{self.enc.opt_model_type.name}/'
        os.makedirs(d,exist_ok=True)
        return d

    def update_model_name(self):
        n = self.name_detail

        self.name = n

    def print_values(self):
        """
        Print all basic_parameters used in the model
        """
        for key, v in self.__dict__.items():
            try:
                print('########', key, '########')
                for key2, vv in v.__dict__.items():
                    print(key2, ':', vv)
            except:
                print(v)

        print('----', flush=True)

    def update_param_grid(self, grid_list, id_comb):
        ind = []
        for l in grid_list:
            t = np.arange(0, len(l[2]))
            ind.append(t.tolist())
        combs = list(itertools.product(*ind))
        print('comb', str(id_comb + 1), '/', str(len(combs)))
        c = combs[id_comb]

        for i, l in enumerate(grid_list):
            self.__dict__[l[0]].__dict__[l[1]] = l[2][c[i]]

    def finalize_parameters(self, verbose=True):
        np.random.seed(self.seed)
        if verbose:
            self.update_model_name()  # automatically create a unique name for the experiment results.
        # create a unique directory name with the basic_parameters of the experiment (the basic_parameters defines the name automatically)
        save_dir = f'{self.model.res_dir}{self.name}/'
        os.makedirs(save_dir, exist_ok=True)
        # print and save the final basic_parameters
        self.save(
            save_dir)  # this save the param object with current configuration. So we will never forget the basic_parameters of each experiment run.

    def save(self, save_dir, file_name='/basic_parameters.p'):
        # simple save function that allows loading of deprecated basic_parameters object
        df = pd.DataFrame(columns=['key', 'value'])

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    temp = pd.DataFrame(data=[str(key) + '_' + str(key2), vv], index=['key', 'value']).T
                    df = pd.concat([df, temp], axis=0)
                    # df = df.append(temp)

            except:
                temp = pd.DataFrame(data=[key, v], index=['key', 'value']).T
                df = pd.concat([df, temp], axis=0)
                # df = df.append(temp)
            df.to_pickle(save_dir + file_name, protocol=4)
        # return df

    def load(self, load_dir, file_name='/basic_parameters.p'):
        # simple load function that allows loading of deprecated basic_parameters object
        df = pd.read_pickle(load_dir + file_name)
        # First check if this is an old pickle version, if so transform it into a df
        if type(df) != pd.DataFrame:
            loaded_par = df
            df = pd.DataFrame(columns=['key', 'value'])
            for key, v in loaded_par.__dict__.items():
                try:
                    for key2, vv in v.__dict__.items():
                        temp = pd.DataFrame(data=[str(key) + '_' + str(key2), vv], index=['key', 'value']).T
                        df = df.append(temp)

                except:
                    temp = pd.DataFrame(data=[key, v], index=['key', 'value']).T
                    df = df.append(temp)

        no_old_version_bug = True

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    t = df.loc[df['key'] == str(key) + '_' + str(key2), 'value']
                    if t.shape[0] == 1:
                        tt = t.values[0]
                        self.__dict__[key].__dict__[key2] = tt
                    else:
                        if no_old_version_bug:
                            no_old_version_bug = False
                            print('#### Loaded basic_parameters object is depreceated, default version will be used')
                        print('Parameter', str(key) + '.' + str(key2), 'not found, using default: ',
                              self.__dict__[key].__dict__[key2])

            except:
                t = df.loc[df['key'] == str(key), 'value']
                if t.shape[0] == 1:
                    tt = t.values[0]
                    self.__dict__[key] = tt
                else:
                    if no_old_version_bug:
                        no_old_version_bug = False
                        print('#### Loaded basic_parameters object is depreceated, default version will be used')
                    print('Parameter', str(key), 'not found, using default: ', self.__dict__[key])

    def get_rf_save_dir(self,temp_dir=False):
        dir_type = 'temp' if temp_dir else 'final'
        save_dir = Constant.MAIN_DIR + f'data/rf/{dir_type}/{self.data.data_to_use_name}/'
        for i, k in enumerate(self.rf.__dict__):
            x = self.rf.__dict__[k]
            if type(x) == list:
                xx=str(len(x))+str(x[0])
            else:
                xx = x
            save_dir += f'{k[0:3]}{xx}'.replace('.','')
        os.makedirs(save_dir,exist_ok=True)
        return save_dir