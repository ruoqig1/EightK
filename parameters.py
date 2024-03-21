# basic_parameters
# Created by Antoine Didisheim, at 06.08.19
# job: store default basic_parameters used throughout the projects in single .py

import datetime
import itertools
import time
from enum import Enum
import numpy as np
import pandas as pd
import socket
import getpass
import os
import hashlib


##################
# Enum
##################

class Framework(Enum):
    PYTORCH = 'pytorch'
    TENSORFLOW = 'tensorflow'

class NewsSource(Enum):
    WSJ = 'wsj'
    EIGHT_LEGAL ='eight_legal'
    EIGHT_LEGAL_ATI ='eight_legal_ati'
    EIGHT_LEGAL_ATI_TRAIN ='eight_legal_ati_train'
    EIGHT_PRESS ='eight_press'
    NEWS_REF ='NEWS_REF'
    NEWS_REF_ON_EIGHT_K ='NEWS_REF_ON_EIGHT_K'
    NEWS_THIRD ='NEWS_THIRD' # third party news
    NEWS_SINGLE ='NEWS_SINGLE' # both third party and ref news, attached to a single stock
    WSJ_ONE_PER_STOCK = 'WSJ_ONE_PER_STOCK'

class PredModel(Enum):
    RIDGE = 'RIDGE'
    LOGIT_EN ='LOGIT_EN'

class Normalisation(Enum):
    NO = 'no'
    RANK ='RANK'
    ZSCORE ='ZSCORE'
    MINMAX ='MINMAX'

class PredictorsCoverage(Enum):
    ALL = 'all'
    COVE_ONLY = 'cove_only'
    ALL_BUT_COV = 'all_but_cove'
    ITEMS_NAMES = 'items'
    ITEMS_NAMES_AND_SIZE = 'items_and_size'


class MLModel:
    RF ='RF'

class VocabularySetTfIdf(Enum):
    ALL = 'All'
    REUTERS_ONLY = 'reuters_only'
    WSJ_ONLY = 'wsj_only'


class OptModelType(Enum):
    OPT_125m ='facebook/opt-125M'
    OPT_350m ='facebook/opt-350M'
    OPT_1b3 ='facebook/opt-1.3b'
    OPT_2b6 ='facebook/opt-2.7b'
    OPT_6b7 ='facebook/opt-6.7b'
    OPT_13b ='facebook/opt-13b'
    OPT_30b ='facebook/opt-30b'
    OPT_66b ='facebook/opt-66b'
    OPT_175b ='facebook/opt-175b'
    BOW1 ='BOW1'



##################
# constant
##################


class Constant:
    if (socket.gethostname() in ['HEC37827','3330L-214940-M']) | ('gadi' in socket.gethostname()):
        # main_dir = '/media/antoine/ssd_ntfs//wsj_openai/'
        MAIN_DIR = './'
        HUGGING_DIR = None
        HUGGING_DIR_TORCH = None
    elif socket.gethostname() in ['rdl-7enbvm.desktop.cloud.unimelb.edu.au']:
        MAIN_DIR = '/mnt/layline/project/eightk/'
        HUGGING_DIR = None
        HUGGING_DIR_TORCH = None
    elif socket.gethostname() in ['Ruoqis-MacBook-Pro.local', 'ravpn-266-2-student-10-8-64-161.uniaccess.unimelb.edu.au',
                                  'Ruoqis-MBP.net']:
        MAIN_DIR = '/Users/ruoqig/punim2039/EightK/'
        HUGGING_DIR = '/Users/ruoqig/punim2039/hugging/'
        HUGGING_DIR_TORCH = '/Users/ruoqig/punim2039/hugging_torch/'
    elif getpass.getuser() in ['ruoqig']:
        MAIN_DIR = '/data/gpfs/projects/punim2119/EightK/'
        HUGGING_DIR = '/data/gpfs/projects/punim2039/hugging/'
        HUGGING_DIR_TORCH = '/data/gpfs/projects/punim2039/hugging_torch/'
    else:
        MAIN_DIR = '/data/gpfs/projects/punim2039/EightK/'
        HUGGING_DIR = '/data/gpfs/projects/punim2039/hugging/'
        HUGGING_DIR_TORCH = '/data/gpfs/projects/punim2039/hugging_torch/'

    HOME_DIR = os.path.expanduser("~")
    EMB_PAPER = os.path.join(HOME_DIR, 'Dropbox/Apps/Overleaf/052-EMB/res/')
    TRI_PAPER = os.path.join(HOME_DIR, 'Dropbox/Apps/Overleaf/053-TRI/res/')
    DROPBOX_COSINE_DATA = os.path.join(HOME_DIR, 'Dropbox/AB-AD_Share/cosine/data/')

    PATH_TO_MODELS_NOW = 'res/model_tf_ati_dec_spartan/'

    COLOR_DUO = ['k', 'b']
    COLOR_TRIO = ['k', 'b','g']
    FRED_API = 'cfc526395a5631650ec0b7ee96b149f4'

    DRAFT_1_CSV_PATH = '/Users/adidisheim/Dropbox/AB-AD_Share/052-EMB/datasets/draft_1/csv/'

    DROP_RES_DIR = '/Users/adidisheim/Dropbox/Apps/Overleaf/EightKEarlyAnalysis/'

    LIST_ITEMS_TO_USE = [1.01, 1.02, 1.03, 1.04, 2.01, 2.03, 2.04, 2.05, 2.06, 3.02, 3.03, 4.01, 4.02, 5.01, 5.02, 5.03, 5.04, 5.05, 5.06, 5.07, 5.08, 6.01, 6.02, 6.03, 6.04, 6.05, 7.01, 8.01]
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

    IS_VM = socket.gethostname()=='rdl-7enbvm.desktop.cloud.unimelb.edu.au'

class BryanGroups:
    accounting_size = ['assets','sales','book_equity','net_income','enterprise_value']
    growth_percentage = [
        'at_gr1',
        'sale_gr1',
        'ca_gr1',
        'nca_gr1',
        'lt_gr1',
        'cl_gr1',
        'ncl_gr1',
        'be_gr1',
        'pstk_gr1',
        'debt_gr1',
        'cogs_gr1',
        'sga_gr1',
        'opex_gr1',
        'at_gr3',
        'nca_gr3',
        'lt_gr3',
        'cl_gr3',
        'ncl_gr3',
        'be_gr3',
        'pstk_gr3',
        'debt_gr3',
        'cogs_gr3',
        'sga_gr3',
        'opex_gr3'
    ]
    growth_changed_scale_by_total_asset = [
        'gp_gr3a',
        'ocf_gr3a',
        'cash_gr3a',
        'inv_gr3a',
        'rec_gr3a',
        'ppeg_gr3a',
        'lti_gr3a',
        'intan_gr3a',
        'debtst_gr3a',
        'ap_gr3a',
        'txp_gr3a',
        'debtlt_gr3a',
        'txditc_gr3a',
        'coa_gr3a',
        'cowc_gr3a',
        'ncoa_gr3a',
        'nncoa_gr3a',
        'oa_gr3a',
        'ol_gr3a',
        'ncoa_gr3a',
        'fna_gr3a',
        'fnl_gr3a',
        'nfna_gr3a',
        'ebitda_gr3a',
        'ope_gr3a',
        'ni_gr3a',
        'dp_gr3a',
        'fcf_gr3a',
        'nwc_gr3a',
        'nix_gr3a',
        'eqnetis_gr3a',
        'dltnetis_gr3a',
        'dstnetis_gr3a',
        'dbnetis_gr3a',
        'netis_gr3a',
        'fincf_gr3a',
        'eqnpo_gr3a',
        'tax_gr3a',
        'div_gr3a',
        'eqbb_gr3a',
        'eqis_gr3a',
        'eqpo_gr3a',
        'capx_gr3a',
    ]

    investment = [
        'capx_at',
        'rd_at'
    ]

    non_recurring_items = [
        'spi_at',
        'xido_at',
        'nri_at'
    ]

    profit_margin = [
        'gp_sale',
        'ebitda_sale',
        'ebit_sale',
        'pi_sale',
        'ni_sale',
        'nix_sale',
        'fcf_sale',
        'ocf_sale'
    ]

    return_on_assets = [
        'gp_at',
        'ebit_at',
        'ebitda_at',
        'fi_at',
        'cop_at'
    ]

    return_on_book_equity = [
        'ope_be',
        'ni_be',
        'nix_be',
        'ocf_be',
        'fcf_be'
    ]

    return_on_invested_capital = [
        'gp_bev',
        'ebitda_bev',
        'ebit_bev',
        'fi_bev',
        'cop_bev'
    ]

    return_on_physical_capital = [
        'gp_ppen',
        'ebitda_ppen',
        'fcf_ppen'
    ]

    issuance = [
        'fincf_at',
        'netis_at',
        'eqnetis_at',
        'eqis_at',
        'dbnetis_at',
        'dltnetis_at',
        'dstnetis_at'
    ]

    equity_payout = [
        'eqnpo_at',
        'eqbb_at',
        'div_at'
    ]

    accruals = [
        'oaccruals_at',
        'oaccruals_ni',
        'taccruals_at',
        'taccruals_ni',
        'noa_at'
    ]

    capitalisation_leverage_ratio = [
        'be_bev',
        'debt_bev',
        'cash_bev',
        'pstk_bev',
        'debtlt_bev',
        'debtst_bev',
        'debt_mev',
        'pstk_mev',
        'debtlt_mev',
        'debtst_mev',
    ]

    finanical_soundness_ratios = [
        'int_debt',
        'int_debtlt',
        'ebitda_debt',
        'profit_cl',
        'ocf_cl',
        'ocf_debt',
        'cash_lt',
        'inv_act',
        'rec_act',
        'debtst_debt',
        'cl_lt',
        'debtlt_debt',
        'opex_at',
        'fcf_ocf',
        'lt_ppen',
        'debtlt_be',
        'nwc_at',
    ]

    solvency_ratios = [
        'debt_at',
        'debt_be',
        'ebit_int'
    ]

    liquidity_ratios = [
        'inv_days',
        'rec_days',
        'ap_days',
        'cash_conversion',
        'cash_cl',
        'caliq_cl',
        'ca_cl',
    ]

    activity_efficency_ratios =[
        'inv_turnover',
        'at_turnover',
        'rec_turnover',
        'ap_turnover',
        'adv_sale',
        'staff_sale',
        'sale_bev',
        'rd_sale',
        'sale_be',
        'div_ni',
        'sale_nwc',
        'tax_pi'
    ]

    balance_sheet_fundamental_to_market_equity = [
        'be_me',
        'at_me',
        'cash_me'
    ]

    income_fundamentals_to_market_equity = [
        'gp_me',
        'ebitda_me',
        'ebit_me',
        'ope_me',
        'ni_me',
        'sale_me',
        'ocf_me',
        'fcf_me',
        'nix_me',
        'cop_me',
        'rd_me'
    ]
    balance_sheet_fundamental_to_entreprise_value = [
        'be_mev',
        'at_mev',
        'cash_mev',
        'bev_mev',
        'ppen_mev'
    ]

    equity_payout_issuance_to_market_equity = [
        'div_me',
        'eqbb_me',
        'eqis_me',
        'eqpo_me',
        'eqnpo_me',
        'eqnetis_me'
    ]

    debt_issuance_to_market_entreprise_value = [
        'dltnetis_mev',
        'dstnetis_mev',
        'dbnetis_mev'
    ]

    firm_payout_issuance_to_market_enterprise_value = [
        'netis_mev'
    ]

    income_fundamentals_to_market_entreprise_value = [
        'gp_mev',
        'ebitda_mev',
        'ebit_mev',
        'sale_mev',
        'ocf_mev',
        'fcf_mev',
        'cop_mev',
        'fincf_mev'
    ]

    new_variables_not_in_hxz = [
        'niq_saleq_std',
        'ni_emp',
        'sale_emp',
        'ni_at',
        'ocf_at',
        'ocf_at_chg1',
        'roeq_be_std',
        'roe_be_std',
        'gpoa_ch5',
        'roe_ch5',
        'roa_ch5',
        'cfoa_ch5',
        'gmar_ch5'
    ]

    new_variables_from_hxz = [
        'cash_at',
        'ni_inc8q',
        'ppeinv_gr1a',
        'lnoa_gr1a',
        'capx_gr1',
        'capx_gr2',
        'capx_gr3',
        'sti_gr1a',
        'niq_be',
        'niq_be_chg1',
        'niq_at',
        'niq_at_chg1',
        'saleq_gr1',
        'rd5_at',
        'age',
        'dsale_dinv',
        'dsale_drec',
        'dgp_dsale',
        'dsale_dsga',
        'saleq_su',
        'niq_su',
        'debt_me',
        'netdebt_me',
        'capex_abn',
        'inv_gr1',
        'be_gr1a',
        'op_at',
        'pi_nix',
        'op_atl1',
        'ope_bel1',
        'gp_atl1',
        'cop_atl1',
        'at_be',
        'ocfq_saleq_std',
        'aliq_at',
        'aliq_mat',
        'tangibility',
        'eq_dur',
        'f_score',
        'o_score',
        'z_score',
        'kz_index',
        'intrinsic_value',
        'ival_me',
        'sale_emp_gr1',
        'emp_gr1',
        'earnings_variability',
        'ni_ar1',
        'ni_ivol'
    ]
    # todo finish from table 7
    # market_based_size_measures =

    @staticmethod
    def get_all_names():
        return [x for x in BryanGroups.__dict__.keys() if ('__' not in x) & ('get_' not in x)]


BRYAN_MAIN_CATEGORIES = {
    "Skewness": [
        "iskew_hxz4_21d",
        "iskew_ff3_21d",
        "iskew_capm_21d",
        "rskew_21d",
        "rmax5_rvol_21d",
        "ret_1_0",
    ],
    "Profitability": [
        "o_score",
        "ebit_sale",
        "f_score",
        "ocf_at",
        "ope_be",
        "ni_be",
        "ebit_bev",
        "niq_be",
        "ope_bel1",
        "turnover_var_126d",
        "dolvol_var_126d",
    ],
    "Low Risk": [
        "betabab_1260d",
        "beta_60m",
        "betadown_252d",
        "beta_dimson_21d",
        "seas_6_10na",
        "zero_trades_126d",
        "turnover_126d",
        "zero_trades_252d",
        "zero_trades_21d",
        "ivol_hxz4_21d",
        "ivol_ff3_21d",
        "ivol_capm_21d",
        "ivol_capm_252d",
        "rmax5_21d",
        "rmax1_21d",
        "rvol_21d",
        "ocfq_saleq_std",
        "earnings_variability",
    ],
    "Value": [
        "eqnetis_at",
        "chcsho_12m",
        "netis_at",
        "fcf_me",
        "eqpo_me",
        "div12m_me",
        "eqnpo_me",
        "eqnpo_12m",
        "bev_mev",
        "at_me",
        "be_me",
        "debt_me",
        "eq_dur",
        "intrinsic_value",  # "ival_me",
        "sale_me",
        "ebitda_mev",
        "ocf_me",
        "ni_me",
    ],
    "Investment": [
        "emp_gr1",
        "aliq_at",
        "be_gr1a",
        "at_gr1",
        "capx_gr1",
        "saleq_gr1",
        "sale_gr1",
        "col_gr1a",
        "inv_gr1a",
        "inv_gr1",
        "coa_gr1a",
        "nncoa_gr1a",
        "ncoa_gr1a",
        "lnoa_gr1a",
        "noa_gr1a",
        "mispricing_mgmt",
        "ppeinv_gr1a",
        "capx_gr3",
        "capx_gr2",
        "sale_gr3",
        "seas_2_5na",
        "ret_60_12",
    ],
    "Seasonality": [
        "coskew_21d",
        "corr_1260d",
        "kz_index",
        "dbnetis_at",
        "lti_gr1a",
        "sti_gr1a",
        "pi_nix",
        "seas_6_10an",
        "seas_11_15an",
        "seas_16_20an",
        "seas_2_5an",
        "seas_11_15na",
    ],
    "Debt Issuance": [
        "noa_at",
        "ncol_gr1a",
        "capex_abn",
        "ni_ar1",
        "nfna_gr1a",
        "fnl_gr1a",
        "debt_gr3",
    ],
    "Size": ["rd_me", "prc", "market_equity", "ami_126d", "dolvol_126d"],
    "Accruals": [
        "taccruals_at",
        "oaccruals_at",
        "cowc_gr1a",
        "taccruals_ni",
        "oaccruals_ni",
        "seas_16_20na",
    ],
    "Low Leverage": [
        "netdebt_me",
        "cash_at",
        "z_score",
        "at_be",
        "rd5_at",
        "rd_sale",
        "aliq_mat",
        "tangibility",
        "ni_ivol",
        "bidaskhl_21d",
        "age",
    ],
    "Profit Growth": [
        "seas_1_1an",
        "ret_12_7",
        "dsale_drec",
        "tax_gr1a",
        "saleq_su",
        "niq_be_chg1",
        "niq_at_chg1",
        "niq_su",
        "ocf_at_chg1",
        "dsale_dinv",
        "sale_emp_gr1",
        "dsale_dsga",
    ],
    "Momentum": [
        "ret_3_1",
        "prc_highprc_252d",
        "seas_1_1na",
        "ret_12_1",
        "ret_9_1",
        "ret_6_1",
        "resff3_6_1",
        "resff3_12_1",
    ],
    "Quality": [
        "qmj_prof",
        "niq_at",
        "mispricing_perf",
        "op_atl1",
        "op_at",
        "cop_atl1",
        "cop_at",
        "qmj_growth",
        "qmj",
        "ni_inc8q",
        "dgp_dsale",
        "qmj_safety",
        "opex_at",
        "at_turnover",
        "sale_bev",
        "gp_atl1",
        "gp_at",
    ],
}




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

class CoveragePredict:
    def __init__(self):
        self.predictors = PredictorsCoverage.ITEMS_NAMES
        self.normalize = Normalisation.ZSCORE
        self.contemp_cov = None # None
        self.model = MLModel.RF
        self.small_sample = False
        self.use_age_and_market_only = None # None

class AbnormalRetParams:
    def __init__(self):
        self.ev_window = 20
        self.gap_window = 50
        self.rolling_window = 100
        self.mkt_col = ['mktrf', 'one']
        self.min_rolling = 70
        self.min_test = 41
    def get_unique_name(self):
        # create the directory
        d = self.__dict__
        s=''
        for k in d.keys():
            if d[k] is not None:
                # we only add to the string name if a parameters is not none. That allows us to keep compatible stuff with old models by adding new parameters with None
                # v= d[k] if type(d[k]) not in [type(np.array([])),type([])] else len(d[k])
                if type(d[k]) in [type(np.array([])),type([])]:
                    v = len(d[k])
                    if v ==1:
                        v = d[k][0]
                else:
                    v = d[k]
                s+= f'{k}{v}'
        return s


class TfIdfParams:
    def __init__(self):
        self.no_below = 1
        self.no_above = 0.99
        self.do_some_filtering = False
        self.vocabulary_list = VocabularySetTfIdf.REUTERS_ONLY



class EncodingParams:
    def __init__(self):
        self.opt_model_type = OptModelType.OPT_125m
        self.news_source = NewsSource.EIGHT_LEGAL
        self.framework = None

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


class GridParams:
    def __init__(self):
        self.year_id = 0

class TrainerParams:
    def __init__(self):
        self.T_train = 360
        self.T_val = 36
        self.testing_window = 1
        self.shrinkage_list = np.linspace(1e-12,10,50)

        self.pred_model = PredModel.RIDGE
        self.norm = Normalisation.ZSCORE
        self.save_ins = False
        self.tnews_only = False
        self.news_filter_training = None # 'news0', 'rtime_nr', 'news0_nr', 'news_with_time', 'news_with_time_nr'
        self.l1_ratio = [0.5]
        self.abny = None # if True, we train the model o nabnormal return

        # this is the number of individual saving chunks.
        # by this we mean the number of individual df contianing some oos performance that will be saved before merged.
        # too big and we risk loosing some processing, too small and we will make a mess of the merging process.
        self.nb_chunks = None
        #
        self.min_nb_chunks_in_cluster = None
        self.use_tf_models = None
        self.batch_size = None
        self.adam_rate = 0.001
        self.patience = 6  # number of epochs to wait before early stopping (5-10 for small models, 10-20 for big models)
        self.monitor_metric = 'loss'  # Metric used to monitor for early stopping (loss or val_auc)
        self.max_epoch = None  # ='loss'
        self.train_on_gpu = None
        self.tensorboard = False

        self.apply_filter = None
        self.filter_on_reuters = None
        self.filter_on_alert = None
        self.filter_on_prn = None
        self.filter_on_cosine = None
        # now we put here the tf model parameters that we have to define

        # for checking the logistic works
        self.sanity_check = None


# store all basic_parameters into a single object
class Params:
    def __init__(self):
        self.name_detail = 'default'
        self.name = ''
        self.use_hash = True
        self.seed = 12345
        self.data = DataParams()
        self.enc = EncodingParams()
        self.train = TrainerParams()
        self.rf = RandomFeaturesParams()
        self.grid = GridParams()
        self.tfidf = TfIdfParams()
        self.model_ran_dir = Constant.MAIN_DIR+'res/model_ran/'
        self.covpred = CoveragePredict()

    def get_coverage_predict_save_dir(self):
        cov_enc = self.dict_to_string_for_dir(self.covpred.__dict__,old_style=True)
        dir_ = Constant.MAIN_DIR+f'res/cov_pred_rf/{cov_enc}/'
        os.makedirs(dir_,exist_ok=True)
        return dir_

    def get_vec_process_dir(self, merged_bow = False, index_permno_only = False):
        # create the directory
        if not merged_bow:
            if self.enc.framework is None:
                save_dir = Constant.MAIN_DIR + f'data/vec_process/{self.enc.opt_model_type.name}/{self.enc.news_source.name}/'
            else:
                save_dir = Constant.MAIN_DIR+f'data/vec_process_news/{self.enc.opt_model_type.name}/'+self.dict_to_string_for_dir(self.enc.__dict__,old_style=True)+'/'
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = Constant.MAIN_DIR + f'data/vec_merged_df/'
            os.makedirs(save_dir, exist_ok=True)
            if index_permno_only:
                save_dir += f'df_{self.enc.opt_model_type.name}_{self.enc.news_source.name}_index.p'
            else:
                save_dir += f'df_{self.enc.opt_model_type.name}_{self.enc.news_source.name}.p'

        return save_dir

    def save_model_params_in_main_file(self):
        os.makedirs(self.model_ran_dir,exist_ok=True)
        # Get current date and time
        now = datetime.datetime.now()
        # Format it into a string suitable for a filename

        k = max([int(x.split('_')[1]) for x in os.listdir(self.model_ran_dir)]+[0])+1
        formatted_date = now.strftime("%Y-%m-%d")+f'_{k}'
        self.save(save_dir=self.model_ran_dir,file_name=formatted_date)
        print('Saved models params in',self.model_ran_dir+formatted_date,flush=True)

    def load_model_params_in_main_file(self,k=None,date=None):
        last_k = max([int(x.split('_')[1]) for x in os.listdir(self.model_ran_dir)])
        k_dict = {int(x.split('_')[1]):x for x in os.listdir(self.model_ran_dir)}
        date_dict = {x.split('_')[0]:x for x in os.listdir(self.model_ran_dir)}

        if (k is None) & (date is None):
            #load the last version
            self.load(load_dir=self.model_ran_dir,file_name=k_dict[last_k])
        if (k is not None):
            self.load(load_dir=self.model_ran_dir,file_name=k_dict[k])
        if (date is not None):
            self.load(load_dir=self.model_ran_dir,file_name=date_dict[date])


    def get_tf_idf_dir(self):
        # create the directory
        s_enc = self.dict_to_string_for_dir(self.enc.__dict__,old_style=True)
        s_tf = self.dict_to_string_for_dir(self.tfidf.__dict__,old_style=True)
        save_dir = self.data.base_data_dir + f'tfidf/{s_enc}/{s_tf}/'
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
    def get_cosine_dir(self,temp=False):
        # create the directory
        s_enc = self.dict_to_string_for_dir(self.enc.__dict__,old_style=True)
        t_enc = self.dict_to_string_for_dir(self.tfidf.__dict__,old_style=True)
        if temp:
            save_dir = self.data.base_data_dir + f'temp_cosine/{s_enc}/{t_enc}/'
        else:
            save_dir = self.data.base_data_dir + f'cosine/{s_enc}/{t_enc}/'
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def get_res_dir(self, temp=True, s=""):
        # create the directory
        s = s if s else self.dict_to_string_for_dir(self.train.__dict__)
        temp_str = '/temp'if temp else ''
        save_dir = Constant.MAIN_DIR + f'res{temp_str}/vec_pred/{s}/{self.enc.opt_model_type.name}/{self.enc.news_source.name}/'
        os.makedirs(save_dir, exist_ok=True)
        return save_dir


    def get_training_dir(self):
        # create the directory
        if self.train.sanity_check is None:
            if self.train.use_tf_models is None:
                save_dir = Constant.MAIN_DIR + f'data/training/{self.enc.opt_model_type.name}/{self.enc.news_source.name}/'
            else:
                save_dir = Constant.MAIN_DIR + f'data/training_tf/{self.enc.opt_model_type.name}/{self.enc.news_source.name}/'
        else:
            save_dir = 'logistic_sanity_check/tf/'
        os.makedirs(save_dir, exist_ok=True)
        return save_dir


    def get_training_norm_dir(self):
        # create the directory
        save_dir = Constant.MAIN_DIR + f'data/training_norm/{self.enc.opt_model_type.name}/{self.enc.news_source.name}/{self.train.norm.name}/'
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

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

    def dict_to_string_for_dir(self, d:dict, old_style =False):
        if (self.use_hash) & (old_style==False):
            valid_params = {k: v for k, v in d.items() if v is not None}
            # Convert the dictionary to a string representation
            param_string = str(valid_params)

            # Create a hash of the string
            hash_object = hashlib.sha256(param_string.encode())
            s = hash_object.hexdigest()
        else:
            # the old version for backward compatibility
            s = ''
            for k in d.keys():
                if d[k] is not None:
                    # we only add to the string name if a parameters is not none. That allows us to keep compatible stuff with old models by adding new parameters with None
                    # v= d[k] if type(d[k]) not in [type(np.array([])),type([])] else len(d[k])
                    if type(d[k]) in [type(np.array([])), type([])]:
                        v = len(d[k])
                        if v == 1:
                            v = d[k][0]
                    else:
                        v = d[k]
                    s += f'{k}{v}'
        return s



