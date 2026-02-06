from __future__ import annotations
import json, math, time
import sys
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import pandas as pd
import numpy as np
from fredapi import Fred
from pandas.tseries.offsets import MonthEnd, MonthBegin, DateOffset

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import (
    DATA_PATH, START_DATE, END_DATE, FRED_API_KEY,
    REFRESH_CACHE, TEMP_DIR, MODEL_TYPE, setup_logger
)
from Train.pipeline_helpers import build_hierarchy_structure
from Load_Data.scrape_bls_schedule import get_future_nfp_dates

logger = setup_logger(__file__, TEMP_DIR)

FRED_EMPLOYMENT_CODES = {
    # LEVEL 0: TOTAL
    "total": "PAYEMS",  # Total Nonfarm Employment (SA)
    "total_nsa": "PAYNSA",  # Total Nonfarm Employment (NSA)
    
    # LEVEL 1: MAJOR DIVISIONS
    "total.private": "USPRIV",  # Total Private Employment (SA)
    "total.private_nsa": "CEU0500000001",  # Total Private Employment (NSA)
    "total.government": "USGOVT",  # Total Government Employment (SA)
    "total.government_nsa": "CEU9000000001",  # Total Government Employment (NSA)
    
    # LEVEL 2: PRIVATE SECTOR BREAKDOWN
    "total.private.goods": "USGOOD",  # Goods-Producing Industries (SA)
    "total.private.goods_nsa": "CEU0600000001",  # Goods-Producing Industries (NSA)
    "total.private.services": "CES0800000001",  # Private Service-Providing (SA)
    "total.private.services_nsa": "CEU0800000001",  # Private Service-Providing (NSA)
    
    # LEVEL 3: GOODS-PRODUCING INDUSTRIES
    "total.private.goods.mining_logging": "USMINE",  # Mining and Logging (SA)
    "total.private.goods.mining_logging_nsa": "CEU1000000001",  # Mining and Logging (NSA)
    "total.private.goods.construction": "USCONS",  # Construction (SA)
    "total.private.goods.construction_nsa": "CEU2000000001",  # Construction (NSA)
    "total.private.goods.manufacturing": "MANEMP",  # Manufacturing (SA)
    "total.private.goods.manufacturing_nsa": "CEU3000000001",  # Manufacturing (NSA)
    
    # LEVEL 4: MINING AND LOGGING BREAKDOWN
    "total.private.goods.mining_logging.logging": "CES1011330001",  # Logging (SA)
    "total.private.goods.mining_logging.logging_nsa": "CEU1011330001",  # Logging (NSA)
    "total.private.goods.mining_logging.mining": "CES1021000001",  # Mining, Quarrying, and Oil and Gas (SA)
    "total.private.goods.mining_logging.mining_nsa": "CEU1021000001",  # Mining, Quarrying, and Oil and Gas (NSA)
    
    # LEVEL 5: MINING BREAKDOWN
    "total.private.goods.mining_logging.mining.oil_gas": "CES1021100001",  # Oil and Gas Extraction (SA)
    "total.private.goods.mining_logging.mining.oil_gas_nsa": "CEU1021100001",  # Oil and Gas Extraction (NSA)
    "total.private.goods.mining_logging.mining.except_oil_gas": "CES1021200001",  # Mining except Oil and Gas (SA)
    "total.private.goods.mining_logging.mining.except_oil_gas_nsa": "CEU1021200001",  # Mining except Oil and Gas (NSA)
    "total.private.goods.mining_logging.mining.support": "CES1021300001",  # Support Activities for Mining (SA)
    "total.private.goods.mining_logging.mining.support_nsa": "CEU1021300001",  # Support Activities for Mining (NSA)
    
    # LEVEL 4: CONSTRUCTION BREAKDOWN
    "total.private.goods.construction.buildings": "CES2023600001",  # Construction of Buildings (SA)
    "total.private.goods.construction.buildings_nsa": "CEU2023600001",  # Construction of Buildings (NSA)
    "total.private.goods.construction.heavy_civil": "CES2023700001",  # Heavy and Civil Engineering (SA)
    "total.private.goods.construction.heavy_civil_nsa": "CEU2023700001",  # Heavy and Civil Engineering (NSA)
    "total.private.goods.construction.specialty": "CES2023800001",  # Specialty Trade Contractors (SA)
    "total.private.goods.construction.specialty_nsa": "CEU2023800001",  # Specialty Trade Contractors (NSA)
    
    # LEVEL 5: CONSTRUCTION BUILDINGS BREAKDOWN
    "total.private.goods.construction.buildings.residential": "CES2023610001",  # Residential Building (SA)
    "total.private.goods.construction.buildings.residential_nsa": "CEU2023610001",  # Residential Building (NSA)
    "total.private.goods.construction.buildings.nonresidential": "CES2023620001",  # Nonresidential Building (SA)
    "total.private.goods.construction.buildings.nonresidential_nsa": "CEU2023620001",  # Nonresidential Building (NSA)
    
    # LEVEL 5: SPECIALTY TRADE CONTRACTORS BREAKDOWN
    #"total.private.goods.construction.specialty.residential": "CES2023800101",  # Residential Specialty Trade (SA)
    #"total.private.goods.construction.specialty.residential_nsa": "CEU2023800101",  # Residential Specialty Trade (NSA)
    #"total.private.goods.construction.specialty.nonresidential": "CES2023800201",  # Nonresidential Specialty Trade (SA)
    #"total.private.goods.construction.specialty.nonresidential_nsa": "CEU2023800201",  # Nonresidential Specialty Trade (NSA)
    
    # LEVEL 4: MANUFACTURING BREAKDOWN
    "total.private.goods.manufacturing.durable": "DMANEMP",  # Durable Goods Manufacturing (SA)
    "total.private.goods.manufacturing.durable_nsa": "CEU3100000001",  # Durable Goods Manufacturing (NSA)
    "total.private.goods.manufacturing.nondurable": "NDMANEMP",  # Nondurable Goods Manufacturing (SA)
    "total.private.goods.manufacturing.nondurable_nsa": "CEU3200000001",  # Nondurable Goods Manufacturing (NSA)
    
    # LEVEL 5: DURABLE GOODS MANUFACTURING
    "total.private.goods.manufacturing.durable.wood": "CES3132100001",  # Wood Products (SA)
    "total.private.goods.manufacturing.durable.wood_nsa": "CEU3132100001",  # Wood Products (NSA)
    "total.private.goods.manufacturing.durable.nonmetallic": "CES3132700001",  # Nonmetallic Mineral Products (SA)
    "total.private.goods.manufacturing.durable.nonmetallic_nsa": "CEU3132700001",  # Nonmetallic Mineral Products (NSA)
    "total.private.goods.manufacturing.durable.primary_metals": "CES3133100001",  # Primary Metals (SA)
    "total.private.goods.manufacturing.durable.primary_metals_nsa": "CEU3133100001",  # Primary Metals (NSA)
    "total.private.goods.manufacturing.durable.fabricated_metals": "CES3133200001",  # Fabricated Metal Products (SA)
    "total.private.goods.manufacturing.durable.fabricated_metals_nsa": "CEU3133200001",  # Fabricated Metal Products (NSA)
    "total.private.goods.manufacturing.durable.machinery": "CES3133300001",  # Machinery (SA)
    "total.private.goods.manufacturing.durable.machinery_nsa": "CEU3133300001",  # Machinery (NSA)
    "total.private.goods.manufacturing.durable.computer_electronic": "CES3133400001",  # Computer and Electronic Products (SA)
    "total.private.goods.manufacturing.durable.computer_electronic_nsa": "CEU3133400001",  # Computer and Electronic Products (NSA)
    "total.private.goods.manufacturing.durable.electrical_equipment": "CES3133500001",  # Electrical Equipment and Appliances (SA)
    "total.private.goods.manufacturing.durable.electrical_equipment_nsa": "CEU3133500001",  # Electrical Equipment and Appliances (NSA)
    "total.private.goods.manufacturing.durable.transportation_equipment": "CES3133600001",  # Transportation Equipment (SA)
    "total.private.goods.manufacturing.durable.transportation_equipment_nsa": "CEU3133600001",  # Transportation Equipment (NSA)
    "total.private.goods.manufacturing.durable.furniture": "CES3133700001",  # Furniture and Related Products (SA)
    "total.private.goods.manufacturing.durable.furniture_nsa": "CEU3133700001",  # Furniture and Related Products (NSA)
    "total.private.goods.manufacturing.durable.miscellaneous": "CES3133900001",  # Miscellaneous Durable Goods (SA)
    "total.private.goods.manufacturing.durable.miscellaneous_nsa": "CEU3133900001",  # Miscellaneous Durable Goods (NSA)
    
    # LEVEL 6: COMPUTER AND ELECTRONIC PRODUCTS BREAKDOWN
    "total.private.goods.manufacturing.durable.computer_electronic.computer_peripheral": "CES3133410001",  # Computer and Peripheral Equipment (SA)
    "total.private.goods.manufacturing.durable.computer_electronic.computer_peripheral_nsa": "CEU3133410001",  # Computer and Peripheral Equipment (NSA)
    "total.private.goods.manufacturing.durable.computer_electronic.communication_equipment": "CES3133420001",  # Communication Equipment (SA)
    "total.private.goods.manufacturing.durable.computer_electronic.communication_equipment_nsa": "CEU3133420001",  # Communication Equipment (NSA)
    "total.private.goods.manufacturing.durable.computer_electronic.semiconductors": "CES3133440001",  # Semiconductors and Electronic Components (SA)
    "total.private.goods.manufacturing.durable.computer_electronic.semiconductors_nsa": "CEU3133440001",  # Semiconductors and Electronic Components (NSA)
    "total.private.goods.manufacturing.durable.computer_electronic.instruments": "CES3133450001",  # Navigational, measuring, electromedical, and control instruments manufacturing (SA)
    "total.private.goods.manufacturing.durable.computer_electronic.instruments_nsa": "CEU3133450001",  # Navigational, measuring, electromedical, and control instruments manufacturing (NSA)
    # Missing Manufacturing and reproducing magnetic and optical media and audio and video equipment manufacturing
    #"total.private.goods.manufacturing.durable.computer_electronic.miscellaneous": None,  # Aggregation of missing codes (SA)
    #"total.private.goods.manufacturing.durable.computer_electronic.miscellaneous_nsa": None,  # Aggregation of missing codes (NSA)
    
    # LEVEL 5: NONDURABLE GOODS MANUFACTURING
    "total.private.goods.manufacturing.nondurable.food": "CES3231100001",  # Food Manufacturing (SA)
    "total.private.goods.manufacturing.nondurable.food_nsa": "CEU3231100001",  # Food Manufacturing (NSA)
    "total.private.goods.manufacturing.nondurable.textile_mills": "CES3231300001",  # Textile Mills (SA)
    "total.private.goods.manufacturing.nondurable.textile_mills_nsa": "CEU3231300001",  # Textile Mills (NSA)
    "total.private.goods.manufacturing.nondurable.textile_products": "CES3231400001",  # Textile Product Mills (SA)
    "total.private.goods.manufacturing.nondurable.textile_products_nsa": "CEU3231400001",  # Textile Product Mills (NSA)
    "total.private.goods.manufacturing.nondurable.apparel": "CES3231500001",  # Apparel (SA)
    "total.private.goods.manufacturing.nondurable.apparel_nsa": "CEU3231500001",  # Apparel (NSA)
    "total.private.goods.manufacturing.nondurable.paper": "CES3232200001",  # Paper and Paper Products (SA)
    "total.private.goods.manufacturing.nondurable.paper_nsa": "CEU3232200001",  # Paper and Paper Products (NSA)
    "total.private.goods.manufacturing.nondurable.printing": "CES3232300001",  # Printing and Related Support (SA)
    "total.private.goods.manufacturing.nondurable.printing_nsa": "CEU3232300001",  # Printing and Related Support (NSA)
    "total.private.goods.manufacturing.nondurable.petroleum_coal": "CES3232400001",  # Petroleum and Coal Products (SA)
    "total.private.goods.manufacturing.nondurable.petroleum_coal_nsa": "CEU3232400001",  # Petroleum and Coal Products (NSA)
    "total.private.goods.manufacturing.nondurable.chemicals": "CES3232500001",  # Chemicals (SA)
    "total.private.goods.manufacturing.nondurable.chemicals_nsa": "CEU3232500001",  # Chemicals (NSA)
    "total.private.goods.manufacturing.nondurable.plastics_rubber": "CES3232600001",  # Plastics and Rubber Products (SA)
    "total.private.goods.manufacturing.nondurable.plastics_rubber_nsa": "CEU3232600001",  # Plastics and Rubber Products (NSA)
    "total.private.goods.manufacturing.nondurable.miscellaneous": "CES3232900001",  # Miscellaneous Nondurable Goods, Beverage, tobacco, and leather and allied product manufacturing (SA)
    "total.private.goods.manufacturing.nondurable.miscellaneous_nsa": "CEU3232900001",  # Miscellaneous Nondurable Goods, Beverage, tobacco, and leather and allied product manufacturing (NSA)
    
    # LEVEL 3: SERVICE-PROVIDING INDUSTRIES
    "total.private.services.trade_transportation_utilities": "USTPU",  # Trade, Transportation, and Utilities (SA)
    "total.private.services.trade_transportation_utilities_nsa": "CEU4000000001",  # Trade, Transportation, and Utilities (NSA)
    "total.private.services.information": "USINFO",  # Information (SA)
    "total.private.services.information_nsa": "CEU5000000001",  # Information (NSA)
    "total.private.services.financial": "USFIRE",  # Financial Activities (SA)
    "total.private.services.financial_nsa": "CEU5500000001",  # Financial Activities (NSA)
    "total.private.services.professional_business": "USPBS",  # Professional and Business Services (SA)
    "total.private.services.professional_business_nsa": "CEU6000000001",  # Professional and Business Services (NSA)
    "total.private.services.education_health": "USEHS",  # Education and Health Services (SA)
    "total.private.services.education_health_nsa": "CEU6500000001",  # Education and Health Services (NSA)
    "total.private.services.leisure_hospitality": "USLAH",  # Leisure and Hospitality (SA)
    "total.private.services.leisure_hospitality_nsa": "CEU7000000001",  # Leisure and Hospitality (NSA)
    "total.private.services.other": "USSERV",  # Other Services (SA)
    "total.private.services.other_nsa": "CEU8000000001",  # Other Services (NSA)
    
    # LEVEL 4: TRADE, TRANSPORTATION, AND UTILITIES BREAKDOWN
    "total.private.services.trade_transportation_utilities.wholesale": "USWTRADE",  # Wholesale Trade (SA)
    "total.private.services.trade_transportation_utilities.wholesale_nsa": "CEU4142000001",  # Wholesale Trade (NSA)
    "total.private.services.trade_transportation_utilities.retail": "USTRADE",  # Retail Trade (SA)
    "total.private.services.trade_transportation_utilities.retail_nsa": "CEU4200000001",  # Retail Trade (NSA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing": "CES4300000001",  # Transportation and Warehousing (SA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing_nsa": "CEU4300000001",  # Transportation and Warehousing (NSA)
    "total.private.services.trade_transportation_utilities.utilities": "CES4422000001",  # Utilities (SA)
    "total.private.services.trade_transportation_utilities.utilities_nsa": "CEU4422000001",  # Utilities (NSA)
    
    # LEVEL 5: WHOLESALE TRADE BREAKDOWN
    "total.private.services.trade_transportation_utilities.wholesale.durable": "CES4142300001",  # Wholesale Trade - Durable Goods (SA)
    "total.private.services.trade_transportation_utilities.wholesale.durable_nsa": "CEU4142300001",  # Wholesale Trade - Durable Goods (NSA)
    "total.private.services.trade_transportation_utilities.wholesale.nondurable": "CES4142400001",  # Wholesale Trade - Nondurable Goods (SA)
    "total.private.services.trade_transportation_utilities.wholesale.nondurable_nsa": "CEU4142400001",  # Wholesale Trade - Nondurable Goods (NSA)
    "total.private.services.trade_transportation_utilities.wholesale.electronic_markets": "CES4142500001",  # Electronic Markets and Agents/Wholesale trade agents and brokers (SA)
    "total.private.services.trade_transportation_utilities.wholesale.electronic_markets_nsa": "CEU4142500001",  # Electronic Markets and Agents/Wholesale trade agents and brokers (NSA)
    
    # LEVEL 5: RETAIL TRADE BREAKDOWN
    "total.private.services.trade_transportation_utilities.retail.motor_vehicles": "CES4244100001",  # Motor Vehicles and Parts Dealers (SA)
    "total.private.services.trade_transportation_utilities.retail.motor_vehicles_nsa": "CEU4244100001",  # Motor Vehicles and Parts Dealers (NSA)
    "total.private.services.trade_transportation_utilities.retail.building_materials": "CES4244400001",  # Building material and garden equipment and supplies dealers (SA)
    "total.private.services.trade_transportation_utilities.retail.building_materials_nsa": "CEU4244400001",  # Building material and garden equipment and supplies dealers (NSA)
    "total.private.services.trade_transportation_utilities.retail.food_beverage": "CES4244500001",  # Food and beverage retailers (SA)
    "total.private.services.trade_transportation_utilities.retail.food_beverage_nsa": "CEU4244500001",  # Food and beverage retailers (NSA)
    # Missing Furniture, home furnishings, electronics, and appliance retailers which is the sum of Furniture and home furnishings retailers and Electronics and appliance retailers
    # "total.private.services.trade_transportation_utilities.retail.furniture_electronics": None,  # Aggregation of Furniture and home furnishings retailers and Electronics and appliance retailers (SA)
    # "total.private.services.trade_transportation_utilities.retail.furniture_electronics_nsa": None,  # Aggregation of Furniture and home furnishings retailers and Electronics and appliance retailers (NSA)
    "total.private.services.trade_transportation_utilities.retail.general_merchandise": "CES4245200001",  # General Merchandise retailers (SA)
    "total.private.services.trade_transportation_utilities.retail.general_merchandise_nsa": "CEU4245200001",  # General Merchandise retailers (NSA)
    "total.private.services.trade_transportation_utilities.retail.health_personal": "CES4244600001",  # Health and Personal Care Stores (SA)
    "total.private.services.trade_transportation_utilities.retail.health_personal_nsa": "CEU4244600001",  # Health and Personal Care Stores (NSA)
    "total.private.services.trade_transportation_utilities.retail.gasoline": "CES4244700001",  # Gasoline Stations (SA)
    "total.private.services.trade_transportation_utilities.retail.gasoline_nsa": "CEU4244700001",  # Gasoline Stations (NSA)
    "total.private.services.trade_transportation_utilities.retail.clothing": "CES4244800001",  # Clothing and Clothing Accessories (SA)
    "total.private.services.trade_transportation_utilities.retail.clothing_nsa": "CEU4244800001",  # Clothing and Clothing Accessories (NSA)
    "total.private.services.trade_transportation_utilities.retail.sporting_goods": "CES4245100001",  # Sporting Goods, Hobby, Book, and Music (SA)
    "total.private.services.trade_transportation_utilities.retail.sporting_goods_nsa": "CEU4245100001",  # Sporting Goods, Hobby, Book, and Music (NSA)

    ## Codes not in BLS Table B1 but available in FRED. 
    # "total.private.services.trade_transportation_utilities.retail.department_stores": "CES4245210001",  # Department Stores (DISCONTINUED Dec 2017) (SA)
    # "total.private.services.trade_transportation_utilities.retail.miscellaneous_nsa": "CEU4245210001",  # Department Stores (DISCONTINUED Dec 2022) (NSA)
    # "total.private.services.trade_transportation_utilities.retail.miscellaneous": "CES4245300001",  # Other Miscellaneous Retailers (SA)
    # "total.private.services.trade_transportation_utilities.retail.miscellaneous_nsa": "CEU4245300001",  # Other Miscellaneous Retailers (NSA)
    # "total.private.services.trade_transportation_utilities.retail.nonstore": "CES4245400001",  # Nonstore Retailers (DISCONTINUED DEC 2022) (SA)
    # "total.private.services.trade_transportation_utilities.retail.nonstore_nsa": "CEU4245400001",  # Nonstore Retailers (DISCONTINUED DEC 2022) (NSA)
    
    # LEVEL 6: RETAIL MOTOR VEHICLES BREAKDOWN
    #"total.private.services.trade_transportation_utilities.retail.motor_vehicles.automobile_dealers": "CES4244110001",  # Automobile Dealers (SA)
    #"total.private.services.trade_transportation_utilities.retail.motor_vehicles.automobile_dealers_nsa": "CEU4244110001",  # Automobile Dealers (NSA)
    # Missing Other motor vehicle dealers and Automotive parts, accessories, and tire retailers, can be aggregated as CES4244100001 - CES4244110001
    #"total.private.services.trade_transportation_utilities.retail.motor_vehicles.miscellaneous": None,  # Aggregation of Other motor vehicle dealers and Automotive parts, accessories, and tire retailers (SA)
    #"total.private.services.trade_transportation_utilities.retail.motor_vehicles.miscellaneous_nsa": None,  # Aggregation of Other motor vehicle dealers and Automotive parts, accessories, and tire retailers (NSA)
    
    # LEVEL 6: FURNITURE AND ELECTRONICS BREAKDOWN
    "total.private.services.trade_transportation_utilities.retail.furniture": "CES4244200001",  # Furniture and Home Furnishings (SA)
    "total.private.services.trade_transportation_utilities.retail.furniture_nsa": "CEU4244200001",  # Furniture and Home Furnishings (NSA)
    "total.private.services.trade_transportation_utilities.retail.electronics": "CES4244300001",  # Electronics and Appliance Stores (SA)
    "total.private.services.trade_transportation_utilities.retail.electronics_nsa": "CEU4244300001",  # Electronics and Appliance Stores (NSA)
    
    # # LEVEL 6: RETAIL GENERAL MERCHANDISE BREAKDOWN (Department Stores is discontinued 2017)
    # "total.private.services.trade_transportation_utilities.retail.general_merchandise.department_stores": "CES4245210001",  # Department Stores (SA)
    # "total.private.services.trade_transportation_utilities.retail.general_merchandise.department_stores_nsa": "CEU4245210001",  # Department Stores (NSA)
    # # Missing Warehouse clubs, supercenters, and other general merchandise retailers, can be aggregated as CES4245200001 - CES4245210001
    # "total.private.services.trade_transportation_utilities.retail.general_merchandise.miscellaneous": None,  # Aggregation of Warehouse clubs, supercenters, and other general merchandise retailers (SA)
    # "total.private.services.trade_transportation_utilities.retail.general_merchandise.miscellaneous_nsa": None,  # Aggregation of Warehouse clubs, supercenters, and other general merchandise retailers (NSA)
    
    # LEVEL 5: TRANSPORTATION AND WAREHOUSING BREAKDOWN
    "total.private.services.trade_transportation_utilities.transportation_warehousing.air": "CES4348100001",  # Air Transportation (SA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.air_nsa": "CEU4348100001",  # Air Transportation (NSA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.rail": "CES4348200001",  # Rail Transportation (SA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.rail_nsa": "CEU4348200001",  # Rail Transportation (NSA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.water": "CES4348300001",  # Water Transportation (SA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.water_nsa": "CEU4348300001",  # Water Transportation (NSA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.truck": "CES4348400001",  # Truck Transportation (SA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.truck_nsa": "CEU4348400001",  # Truck Transportation (NSA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.transit": "CES4348500001",  # Transit and Ground Passenger (SA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.transit_nsa": "CEU4348500001",  # Transit and Ground Passenger (NSA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.pipeline": "CES4348600001",  # Pipeline Transportation (SA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.pipeline_nsa": "CEU4348600001",  # Pipeline Transportation (NSA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.scenic": "CES4348700001",  # Scenic and Sightseeing Transportation (SA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.scenic_nsa": "CEU4348700001",  # Scenic and Sightseeing Transportation (NSA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.support": "CES4348800001",  # Support Activities for Transportation (SA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.support_nsa": "CEU4348800001",  # Support Activities for Transportation (NSA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.couriers": "CES4349200001",  # Couriers and Messengers (SA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.couriers_nsa": "CEU4349200001",  # Couriers and Messengers (NSA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.warehousing": "CES4349300001",  # Warehousing and Storage (SA)
    "total.private.services.trade_transportation_utilities.transportation_warehousing.warehousing_nsa": "CEU4349300001",  # Warehousing and Storage (NSA)
    
    # LEVEL 4: INFORMATION BREAKDOWN
    "total.private.services.information.motion_picture": "CES5051200001",  # Motion Picture and Sound Recording (SA)
    "total.private.services.information.motion_picture_nsa": "CEU5051200001",  # Motion Picture and Sound Recording (NSA)
    "total.private.services.information.publishing": "CES5051100001",  # Publishing Industries (SA)
    "total.private.services.information.publishing_nsa": "CEU5051100001",  # Publishing Industries (NSA)
    "total.private.services.information.broadcasting": "CES5051500001",  # Broadcasting except Internet (SA)
    "total.private.services.information.broadcasting_nsa": "CEU5051500001",  # Broadcasting except Internet (NSA)
    "total.private.services.information.telecommunications": "CES5051700001",  # Telecommunications (SA)
    "total.private.services.information.telecommunications_nsa": "CEU5051700001",  # Telecommunications (NSA)
    "total.private.services.information.data_processing": "CES5051800001",  # Computing Infrastructure Providers, Data Processing, Web Hosting, and Related Services / Data Processing and Hosting (SA)
    "total.private.services.information.data_processing_nsa": "CEU5051800001",  # Computing Infrastructure Providers, Data Processing, Web Hosting, and Related Services / Data Processing and Hosting (NSA)
    "total.private.services.information.other": "CES5051900001",  # Web Search Portals, Libraries, Archives, and Other Information / Other Information Services (SA)
    "total.private.services.information.other_nsa": "CEU5051900001",  # Web Search Portals, Libraries, Archives, and Other Information / Other Information Services (NSA)
    
    # LEVEL 4: FINANCIAL ACTIVITIES BREAKDOWN
    "total.private.services.financial.finance_insurance": "CES5552000001",  # Finance and Insurance (SA)
    "total.private.services.financial.finance_insurance_nsa": "CEU5552000001",  # Finance and Insurance (NSA)
    "total.private.services.financial.real_estate": "CES5553000001",  # Real Estate and Rental and Leasing (SA)
    "total.private.services.financial.real_estate_nsa": "CEU5553000001",  # Real Estate and Rental and Leasing (NSA)
    
    # LEVEL 5: FINANCE AND INSURANCE BREAKDOWN
    "total.private.services.financial.finance_insurance.monetary_authorities": "CES5552100001",  # Monetary Authorities - Central Bank (SA)
    "total.private.services.financial.finance_insurance.monetary_authorities_nsa": "CEU5552100001",  # Monetary Authorities - Central Bank (NSA)
    "total.private.services.financial.finance_insurance.credit_intermediation": "CES5552200001",  # Credit Intermediation and Related Activities (SA)
    "total.private.services.financial.finance_insurance.credit_intermediation_nsa": "CEU5552200001",  # Credit Intermediation and Related Activities (NSA)
    "total.private.services.financial.finance_insurance.securities": "CES5552300001",  # Securities, Commodity Contracts, and Investments (SA)
    "total.private.services.financial.finance_insurance.securities_nsa": "CEU5552300001",  # Securities, Commodity Contracts, and Investments (NSA)
    "total.private.services.financial.finance_insurance.insurance": "CES5552400001",  # Insurance Carriers and Related Activities (SA)
    "total.private.services.financial.finance_insurance.insurance_nsa": "CEU5552400001",  # Insurance Carriers and Related Activities (NSA)
    
    # LEVEL 6: CREDIT INTERMEDIATION BREAKDOWN
    #"total.private.services.financial.finance_insurance.credit_intermediation.depository": "CES5552210001",  # Depository Credit Intermediation (SA)
    #"total.private.services.financial.finance_insurance.credit_intermediation.depository_nsa": "CEU5552210001",  # Depository Credit Intermediation (NSA)
    # Missing Nondepository credit intermediation and Activities related to credit intermediation, can be aggregated as CES5552200001 - CES5552210001
    #"total.private.services.financial.finance_insurance.credit_intermediation.miscellaneous": None,  # Aggregation of Nondepository credit intermediation and Activities related to credit intermediation (SA)
    #"total.private.services.financial.finance_insurance.credit_intermediation.miscellaneous_nsa": None,  # Aggregation of Nondepository credit intermediation and Activities related to credit intermediation (NSA)
    
    # LEVEL 7: DEPOSITORY CREDIT BREAKDOWN
    #"total.private.services.financial.finance_insurance.credit_intermediation.depository.commercial_banking": "CES5552211001",  # Commercial Banking (SA)
    #"total.private.services.financial.finance_insurance.credit_intermediation.depository.commercial_banking_nsa": "CEU5552211001",  # Commercial Banking (NSA)
    # Missing other industries, not shown in BLS or FRED, can be aggregated as CES5552210001 - CES5552211001
    #"total.private.services.financial.finance_insurance.credit_intermediation.depository.miscellaneous": None,  # Aggregation of other industries (SA)
    #"total.private.services.financial.finance_insurance.credit_intermediation.depository.miscellaneous_nsa": None,  # Aggregation of other industries (NSA)
    
    # LEVEL 5: REAL ESTATE BREAKDOWN
    "total.private.services.financial.real_estate.real_estate_only": "CES5553100001",  # Real Estate (SA)
    "total.private.services.financial.real_estate.real_estate_only_nsa": "CEU5553100001",  # Real Estate (NSA)
    "total.private.services.financial.real_estate.rental_leasing": "CES5553200001",  # Rental and Leasing Services (SA)
    "total.private.services.financial.real_estate.rental_leasing_nsa": "CEU5553200001",  # Rental and Leasing Services (NSA)
    "total.private.services.financial.real_estate.lessors_intangible": "CES5553300001",  # Lessors of Nonfinancial Intangible Assets (SA)
    "total.private.services.financial.real_estate.lessors_intangible_nsa": "CEU5553300001",  # Lessors of Nonfinancial Intangible Assets (NSA)
    
    # LEVEL 4: PROFESSIONAL AND BUSINESS SERVICES BREAKDOWN
    "total.private.services.professional_business.professional_technical": "CES6054000001",  # Professional and Technical Services (SA)
    "total.private.services.professional_business.professional_technical_nsa": "CEU6054000001",  # Professional and Technical Services (NSA)
    "total.private.services.professional_business.management_companies": "CES6055000001",  # Management of Companies and Enterprises (SA)
    "total.private.services.professional_business.management_companies_nsa": "CEU6055000001",  # Management of Companies and Enterprises (NSA)
    "total.private.services.professional_business.admin_waste": "CES6056000001",  # Administrative and support and waste management and remediation services (SA)
    "total.private.services.professional_business.admin_waste_nsa": "CEU6056000001",  # Administrative and support and waste management and remediation services (NSA)

    # LEVEL 5: PROFESSIONAL AND TECHNICAL SERVICES BREAKDOWN
    "total.private.services.professional_business.professional_technical.legal": "CES6054110001",  # Legal Services (SA)
    "total.private.services.professional_business.professional_technical.legal_nsa": "CEU6054110001",  # Legal Services (NSA)
    "total.private.services.professional_business.professional_technical.accounting": "CES6054120001",  # Accounting, Tax Preparation, Bookkeeping, and Payroll Services (SA)
    "total.private.services.professional_business.professional_technical.accounting_nsa": "CEU6054120001",  # Accounting, Tax Preparation, Bookkeeping, and Payroll Services (NSA)
    "total.private.services.professional_business.professional_technical.architectural": "CES6054130001",  # Architectural, Engineering, and Related Services (SA)
    "total.private.services.professional_business.professional_technical.architectural_nsa": "CEU6054130001",  # Architectural, Engineering, and Related Services (NSA)
    "total.private.services.professional_business.professional_technical.computer_systems": "CES6054150001",  # Computer Systems Design (SA)
    "total.private.services.professional_business.professional_technical.computer_systems_nsa": "CEU6054150001",  # Computer Systems Design (NSA)
    "total.private.services.professional_business.professional_technical.consulting": "CES6054160001",  # Management, Scientific, and Technical Consulting Services (SA)
    "total.private.services.professional_business.professional_technical.consulting_nsa": "CEU6054160001",  # Management, Scientific, and Technical Consulting Services (NSA)
    # Missing Scientific research and development services, Advertising and related services, and Other professional, scientific, and technical services, can be aggregated as CES6054000001 - all other children
    #"total.private.services.professional_business.professional_technical.miscellaneous": None,  # Aggregation of Scientific research and development services, Advertising and related services, and Other professional, scientific, and technical services (SA)
    #"total.private.services.professional_business.professional_technical.miscellaneous_nsa": None, # Aggregation of Scientific research and development services, Advertising and related services, and Other professional, scientific, and technical services (NSA)
    
    # LEVEL 5: ADMINISTRATIVE AND WASTE SERVICES BREAKDOWN
    "total.private.services.professional_business.admin_waste.admin_support": "CES6056100001",  # Administrative and Support Services (SA)
    "total.private.services.professional_business.admin_waste.admin_support_nsa": "CEU6056100001",  # Administrative and Support Services (NSA)
    "total.private.services.professional_business.admin_waste.waste_management": "CES6056200001",  # Waste Management and Remediation (SA)
    "total.private.services.professional_business.admin_waste.waste_management_nsa": "CEU6056200001",  # Waste Management and Remediation (NSA)
    
    # LEVEL 6: ADMINISTRATIVE AND SUPPORT SERVICES BREAKDOWN
    #"total.private.services.professional_business.admin_waste.admin_support.employment": "CES6056130001",  # Employment Services (SA)
    #"total.private.services.professional_business.admin_waste.admin_support.employment_nsa": "CEU6056130001",  # Employment Services (NSA)
    #"total.private.services.professional_business.admin_waste.admin_support.business_support": "CES6056140001",  # Business Support Services (SA)
    #"total.private.services.professional_business.admin_waste.admin_support.business_support_nsa": "CEU6056140001",  # Business Support Services (NSA)
    #"total.private.services.professional_business.admin_waste.admin_support.buildings_dwellings": "CES6056170001",  # Services to Buildings and Dwellings (SA)
    #"total.private.services.professional_business.admin_waste.admin_support.buildings_dwellings_nsa": "CEU6056170001",  # Services to Buildings and Dwellings (NSA)
    # Missing Office administrative services, Facilities support services, Travel arrangement and reservation services, Investigation and security services, and Other support services, can be aggregated as CES6056100001 - all other children
    #"total.private.services.professional_business.admin_waste.admin_support.miscellaneous": None,  # Aggregation of Office administrative services, Facilities support services, Travel arrangement and reservation services, Investigation and security services, and Other support services (SA)
    #"total.private.services.professional_business.admin_waste.admin_support.miscellaneous_nsa": None,  # Aggregation of Office administrative services, Facilities support services, Travel arrangement and reservation services, Investigation and security services, and Other support services (NSA)
    
    # LEVEL 7: EMPLOYMENT SERVICES BREAKDOWN
    #"total.private.services.professional_business.admin_waste.admin_support.employment.temporary_help": "TEMPHELPS",  # Temporary Help Services (SA)
    #"total.private.services.professional_business.admin_waste.admin_support.employment.temporary_help_nsa": "TEMPHELPN",  # Temporary Help Services (NSA)
    # Missing other industries, not shown in BLS or FRED, can be aggregated as CES6056130001 - TEMPHELPS
    #"total.private.services.professional_business.admin_waste.admin_support.employment.miscellaneous": None, # Aggregation of other industries (SA)
    #"total.private.services.professional_business.admin_waste.admin_support.employment.miscellaneous_nsa": None,  # Aggregation of other industries (NSA)
        
    # LEVEL 4: EDUCATION AND HEALTH SERVICES BREAKDOWN
    "total.private.services.education_health.education": "CES6561000001",  # Private Educational Services (SA)
    "total.private.services.education_health.education_nsa": "CEU6561000001",  # Private Educational Services (NSA)
    "total.private.services.education_health.health_social": "CES6562000001",  # Health Care and Social Assistance (SA)
    "total.private.services.education_health.health_social_nsa": "CEU6562000001",  # Health Care and Social Assistance (NSA)
    
    # LEVEL 5: HEALTH CARE AND SOCIAL ASSISTANCE BREAKDOWN
    "total.private.services.education_health.health_social.health_care": "CES6562000101",  # Health Care (SA)
    "total.private.services.education_health.health_social.health_care_nsa": "CEU6562000101",  # Health Care (NSA)
    "total.private.services.education_health.health_social.social_assistance": "CES6562400001",  # Social Assistance (SA)
    "total.private.services.education_health.health_social.social_assistance_nsa": "CEU6562400001",  # Social Assistance (NSA)
    
    # LEVEL 6: HEALTH CARE BREAKDOWN
    "total.private.services.education_health.health_social.health_care.ambulatory": "CES6562100001",  # Ambulatory Health Care Services (SA)
    "total.private.services.education_health.health_social.health_care.ambulatory_nsa": "CEU6562100001",  # Ambulatory Health Care Services (NSA)
    "total.private.services.education_health.health_social.health_care.hospitals": "CES6562200001",  # Hospitals (SA)
    "total.private.services.education_health.health_social.health_care.hospitals_nsa": "CEU6562200001",  # Hospitals (NSA)
    "total.private.services.education_health.health_social.health_care.nursing": "CES6562300001",  # Nursing and Residential Care Facilities (SA)
    "total.private.services.education_health.health_social.health_care.nursing_nsa": "CEU6562300001",  # Nursing and Residential Care Facilities (NSA)
    
    # LEVEL 7: AMBULATORY HEALTH CARE BREAKDOWN
    #"total.private.services.education_health.health_social.health_care.ambulatory.physicians": "CES6562110001",  # Offices of Physicians (SA)
    #"total.private.services.education_health.health_social.health_care.ambulatory.physicians_nsa": "CEU6562110001",  # Offices of Physicians (NSA)
    #"total.private.services.education_health.health_social.health_care.ambulatory.outpatient": "CES6562140001",  # Outpatient Care Centers (SA)
    #"total.private.services.education_health.health_social.health_care.ambulatory.outpatient_nsa": "CEU6562140001",  # Outpatient Care Centers (NSA)
    #"total.private.services.education_health.health_social.health_care.ambulatory.home_health": "CES6562160001",  # Home Health Care Services (SA)
    #"total.private.services.education_health.health_social.health_care.ambulatory.home_health_nsa": "CEU6562160001",  # Home Health Care Services (NSA)
    # Missing Offices of Dentists, Offices of Other Health Practitioners, Medical and Diagnostic Laboratories, and Other Ambulatory Health Care Services, can be aggregated as CES6562100001 - all other children
    #"total.private.services.education_health.health_social.health_care.ambulatory.miscellaneous": None,  # Aggregation of Offices of Dentists, Offices of Other Health Practitioners, Medical and Diagnostic Laboratories, and Other Ambulatory Health Care Services (SA)
    #"total.private.services.education_health.health_social.health_care.ambulatory.miscellaneous_nsa": None,  # Aggregation of Offices of Dentists, Offices of Other Health Practitioners, Medical and Diagnostic Laboratories, and Other Ambulatory Health Care Services (NSA)
    
    # LEVEL 7: NURSING AND RESIDENTIAL CARE BREAKDOWN
    #total.private.services.education_health.health_social.health_care.nursing.nursing_care": "CES6562310001",  # Nursing Care Facilities (SA)
    #"total.private.services.education_health.health_social.health_care.nursing.nursing_care_nsa": "CEU6562310001",  # Nursing Care Facilities (NSA)
    # Missing Residential intellectual and developmental disability, mental health, and substance abuse facilities, Continuing care retirement communities and assisted living facilities for the elderly, and Other Residential Care Facilities, can be aggregated as CES6562300001 - all other children
    #"total.private.services.education_health.health_social.health_care.nursing.miscellaneous": None,  # Aggregation of Residential intellectual and developmental disability, mental health, and substance abuse facilities, Continuing care retirement communities and assisted living facilities for the elderly, and Other Residential Care Facilities (SA)
    #"total.private.services.education_health.health_social.health_care.nursing.miscellaneous_nsa": None,  # Aggregation of Residential intellectual and developmental disability, mental health, and substance abuse facilities, Continuing care retirement communities and assisted living facilities for the elderly, and Other Residential Care Facilities (NSA)
    
    # LEVEL 6: SOCIAL ASSISTANCE BREAKDOWN
    #"total.private.services.education_health.health_social.social_assistance.child_care": "CES6562440001",  # Child Day Care Services (SA)
    #"total.private.services.education_health.health_social.social_assistance.child_care_nsa": "CEU6562440001",  # Child Day Care Services (NSA)
    # Missing Individual and family services, Community food and housing, and emergency and other relief services, and Vocational rehabilitation services, can be aggregated as CES6562400001 - all other children
    #"total.private.services.education_health.health_social.social_assistance.miscellaneous": None,  # Aggregation of Individual and family services, Community food and housing, and emergency and other relief services, and Vocational rehabilitation services (SA)
    #"total.private.services.education_health.health_social.social_assistance.miscellaneous_nsa": None,  # Aggregation of Individual and family services, Community food and housing, and emergency and other relief services, and Vocational rehabilitation services (NSA)
    
    # LEVEL 4: LEISURE AND HOSPITALITY BREAKDOWN
    "total.private.services.leisure_hospitality.arts": "CES7071000001",  # Arts, Entertainment, and Recreation (SA)
    "total.private.services.leisure_hospitality.arts_nsa": "CEU7071000001",  # Arts, Entertainment, and Recreation (NSA)
    "total.private.services.leisure_hospitality.accommodation_food": "CES7072000001",  # Accommodation and Food Services (SA)
    "total.private.services.leisure_hospitality.accommodation_food_nsa": "CEU7072000001",  # Accommodation and Food Services (NSA)
    
    # LEVEL 5: ARTS, ENTERTAINMENT, AND RECREATION BREAKDOWN
    "total.private.services.leisure_hospitality.arts.performing": "CES7071100001",  # Performing Arts and Spectator Sports (SA)
    "total.private.services.leisure_hospitality.arts.performing_nsa": "CEU7071100001",  # Performing Arts and Spectator Sports (NSA)
    "total.private.services.leisure_hospitality.arts.museums": "CES7071200001",  # Museums, Historical Sites (SA)
    "total.private.services.leisure_hospitality.arts.museums_nsa": "CEU7071200001",  # Museums, Historical Sites (NSA)
    "total.private.services.leisure_hospitality.arts.amusements": "CES7071300001",  # Amusements, Gambling, and Recreation (SA)
    "total.private.services.leisure_hospitality.arts.amusements_nsa": "CEU7071300001",  # Amusements, Gambling, and Recreation (NSA)
    
    # LEVEL 5: ACCOMMODATION AND FOOD SERVICES BREAKDOWN
    "total.private.services.leisure_hospitality.accommodation_food.accommodation": "CES7072100001",  # Accommodation (SA)
    "total.private.services.leisure_hospitality.accommodation_food.accommodation_nsa": "CEU7072100001",  # Accommodation (NSA)
    "total.private.services.leisure_hospitality.accommodation_food.food_services": "CES7072200001",  # Food Services and Drinking Places (SA)
    "total.private.services.leisure_hospitality.accommodation_food.food_services_nsa": "CEU7072200001",  # Food Services and Drinking Places (NSA)
    
    # LEVEL 4: OTHER SERVICES BREAKDOWN
    "total.private.services.other.repair": "CES8081100001",  # Repair and Maintenance (SA)
    "total.private.services.other.repair_nsa": "CEU8081100001",  # Repair and Maintenance (NSA)
    "total.private.services.other.personal": "CES8081200001",  # Personal and Laundry Services (SA)
    "total.private.services.other.personal_nsa": "CEU8081200001",  # Personal and Laundry Services (NSA)
    "total.private.services.other.religious": "CES8081300001",  # Religious, Grantmaking, Civic, Professional, and Similar Organizations (SA)
    "total.private.services.other.religious_nsa": "CEU8081300001",  # Religious, Grantmaking, Civic, Professional, and Similar Organizations (NSA)
    
    # LEVEL 2: GOVERNMENT BREAKDOWN
    "total.government.federal": "CES9091000001",  # Federal Government (SA)
    "total.government.federal_nsa": "CEU9091000001",  # Federal Government (NSA)
    "total.government.state": "CES9092000001",  # State Government (SA)
    "total.government.state_nsa": "CEU9092000001",  # State Government (NSA)
    "total.government.local": "CES9093000001",  # Local Government (SA)
    "total.government.local_nsa": "CEU9093000001",  # Local Government (NSA)
    
    # LEVEL 3: FEDERAL GOVERNMENT BREAKDOWN
    "total.government.federal.except_postal": "CES9091100001",  # Federal except U.S. Postal Service (SA)
    "total.government.federal.except_postal_nsa": "CEU9091100001",  # Federal except U.S. Postal Service (NSA)
    "total.government.federal.postal": "CES9091912001",  # U.S. Postal Service (SA)
    "total.government.federal.postal_nsa": "CEU9091912001",  # U.S. Postal Service (NSA)
    
    # LEVEL 3: STATE GOVERNMENT BREAKDOWN
    "total.government.state.education": "CES9092161101",  # State Government Education (SA)
    "total.government.state.education_nsa": "CEU9092161101",  # State Government Education (NSA)
    "total.government.state.excluding_education": "CES9092200001",  # State Government excluding Education (SA)
    "total.government.state.excluding_education_nsa": "CEU9092200001",  # State Government excluding Education (NSA)
    
    # LEVEL 3: LOCAL GOVERNMENT BREAKDOWN
    "total.government.local.education": "CES9093161101",  # Local Government Education (SA)
    "total.government.local.education_nsa": "CEU9093161101",  # Local Government Education (NSA)
    "total.government.local.excluding_education": "CES9093200001",  # Local Government excluding Education (SA)
    "total.government.local.excluding_education_nsa": "CEU9093200001",  # Local Government excluding Education (NSA)
    }

TOTAL_UIDS = ["total_nsa", "total"]

FRED_ROOT = (DATA_PATH / "fred_data").resolve()
DECADES_ROOT = FRED_ROOT / "decades"
REPORT_FILE = FRED_ROOT / "report.json"
NFP_target_DIR = (DATA_PATH / "NFP_target").resolve()  

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _decade_str(year: int) -> str:
    return f"{(year // 10) * 10}s"  

def month_file_path(as_of: pd.Timestamp) -> Path:
    out_dir = DECADES_ROOT / _decade_str(as_of.year) / f"{as_of.year:04d}"
    _ensure_dir(out_dir)
    return out_dir / f"{as_of.strftime('%Y-%m')}.parquet"

def _write_parquet(df: pd.DataFrame, path: Path, dict_cols: Optional[list[str]] = None) -> None:
    _ensure_dir(path.parent)
    df.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        compression="zstd",
        compression_level=3,
        use_dictionary=dict_cols or []
    )

def _align_audit_to_total_calendar(
    audit: pd.DataFrame,
    total_uid: str = "total_nsa",
) -> pd.DataFrame:
    """
    Align all series' realtime_start to the total_nsa release calendar
    on a per-date, per-vintage-order basis.
    """
    if audit.empty or "unique_id" not in audit.columns:
        return audit

    if total_uid not in audit["unique_id"].unique():
        return audit

    total = audit[audit["unique_id"] == total_uid].copy()
    if total.empty:
        return audit

    total = total.sort_values(["ds", "realtime_start"]).copy()
    total["vintage_idx"] = total.groupby("ds").cumcount()
    calendar = total[["ds", "vintage_idx", "realtime_start"]].rename(
        columns={"realtime_start": "total_realtime_start"}
    )

    aligned = audit.sort_values(["unique_id", "ds", "realtime_start"]).copy()
    aligned["vintage_idx"] = aligned.groupby(["unique_id", "ds"]).cumcount()
    aligned = aligned.merge(calendar, on=["ds", "vintage_idx"], how="left")
    aligned["realtime_start"] = aligned["total_realtime_start"].combine_first(aligned["realtime_start"])
    aligned = aligned.drop(columns=["vintage_idx", "total_realtime_start"])

    aligned["realtime_start"] = pd.to_datetime(aligned["realtime_start"])
    aligned["unique_id"] = aligned["unique_id"].astype("category")
    return aligned

# --- DATE CALCULATION HELPERS ---

def get_first_friday_of_month(dates):
    """
    Returns the 1st Friday of the month for the dates provided.

    Args:
        dates: pd.Series of dates or a single pd.Timestamp

    Returns:
        pd.Series or pd.Timestamp with first Friday of month
    """
    # Handle single Timestamp
    if isinstance(dates, pd.Timestamp):
        start = dates.to_period('M').to_timestamp()
        days_to_add = (4 - start.dayofweek + 7) % 7
        return start + pd.Timedelta(days=days_to_add)

    # Handle Series - Ensure index is preserved by working on the Series directly
    # Snap to start of month using Series methods
    starts = dates.dt.to_period('M').dt.to_timestamp()

    # Calculate days to add to reach Friday (weekday 4)
    # (4 - dow + 7) % 7
    days_to_add = (4 - starts.dt.dayofweek + 7) % 7
    return starts + pd.to_timedelta(days_to_add, unit='D')

def impute_target_release_date_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    SIMPLE LOGIC (For NFP Target "First Release" files):
    Imputes release dates using First Friday of next month rule with different logic based on data quality era:
    
    PRE-2009 (lower data quality):
    - Impute if release_date is missing OR > 14 days after month-end
    
    POST-2009 (higher data quality):
    - Only impute if release_date is explicitly NaN (missing)
    """
    if "release_date" not in df.columns and "realtime_start" in df.columns:
        df = df.rename(columns={"realtime_start": "release_date"})

    # Logic: First Friday of Next Month
    next_month = df["ds"] + MonthBegin(1)
    imputed_dates = get_first_friday_of_month(next_month)

    month_end = df["ds"] + MonthEnd(0)
    threshold = month_end + pd.Timedelta(days=14)

    cutoff_date = pd.Timestamp("2009-01-01")

    # PRE-2009: Impute if missing OR > 14 days after month-end (old logic for lower quality data)
    mask_pre_2009 = (df["release_date"].isna() | (df["release_date"] > threshold)) & (df["ds"] < cutoff_date)
    
    # POST-2009: Only impute if explicitly missing (preserve legitimate delays)
    mask_post_2009 = df["release_date"].isna() & (df["ds"] >= cutoff_date)
    
    # Combine masks
    mask = mask_pre_2009 | mask_post_2009

    # Ensure alignment by using the index
    df.loc[mask, "release_date"] = imputed_dates[mask]

    df["release_date"] = pd.to_datetime(df["release_date"])
    return df

def impute_target_release_date_complex(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    COMPLEX LOGIC (For NFP Target "Last Release" files):
    Applies Options 1/2/3 based on relationship between Data Date, Snapshot Date, and ALFRED Release Date.
    Only imputes for dates before 2009-01-01. After 2009, uses actual FRED data.
    """
    if "release_date" not in df.columns and "realtime_start" in df.columns:
        df = df.rename(columns={"realtime_start": "release_date"})

    cutoff_date = pd.Timestamp("2009-01-01")

    # 1. Identify rows that need imputation
    # Condition: Missing OR > 12 months after data date
    one_year_later = df["ds"] + DateOffset(years=1)

    # Only consider imputation for dates before 2009-01-01
    needs_imputation = ((df["release_date"].isna()) | (df["release_date"] > one_year_later)) & (df["ds"] < cutoff_date)

    if not needs_imputation.any():
        df["release_date"] = pd.to_datetime(df["release_date"])
        return df

    # Prepare Candidate Dates

    # Candidate A: First Friday of NEXT YEAR'S February
    next_year_feb_str = (df["ds"] + DateOffset(years=1)).dt.year.astype(str) + "-02-01"
    next_year_feb = pd.to_datetime(next_year_feb_str)
    cand_feb_next_year = get_first_friday_of_month(next_year_feb)

    # Candidate B1: First Friday of M+1
    cand_m1 = get_first_friday_of_month(df["ds"] + DateOffset(months=1))
    # Candidate B2: First Friday of M+2
    cand_m2 = get_first_friday_of_month(df["ds"] + DateOffset(months=2))
    # Candidate B3: First Friday of M+3
    cand_m3 = get_first_friday_of_month(df["ds"] + DateOffset(months=3))

    # --- LOGIC GATES ---

    # Gate 1: Snapshot is > 1 year after data date
    # OR
    # Gate 2: Snapshot includes next Feb (Snapshot >= cand_feb_next_year)
    # ACTION: Use cand_feb_next_year
    gate_1_2 = (snapshot_date > one_year_later) | (snapshot_date >= cand_feb_next_year)

    # Gate 3: Snapshot is closer (Option 3 logic)
    gate_3 = (~gate_1_2)

    # Apply Option 1 & 2
    mask_1_2 = needs_imputation & gate_1_2
    df.loc[mask_1_2, "release_date"] = cand_feb_next_year[mask_1_2]

    # Apply Option 3
    if gate_3.any():
        mask_3 = needs_imputation & gate_3

        # Create a mini dataframe to do the row-wise max logic efficiently
        candidates = pd.DataFrame({
            "m1": cand_m1[mask_3],
            "m2": cand_m2[mask_3],
            "m3": cand_m3[mask_3]
        }, index=df.index[mask_3])

        # Filter: Set dates > snapshot_date to NaT
        candidates[candidates > snapshot_date] = pd.NaT

        # Take the Max (latest valid date)
        best_date = candidates.max(axis=1)

        # Assign
        df.loc[mask_3, "release_date"] = best_date

    df["release_date"] = pd.to_datetime(df["release_date"])
    return df

def calculate_complex_release_date(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp
) -> Tuple[pd.DataFrame, int]:
    """
    COMPLEX LOGIC (For Snapshots):
    Applies Option 1, 2, or 3 based on relationship between Data Date, Snapshot Date, and ALFRED Release Date.
    Only imputes for dates before 2009-01-01. After 2009, uses actual FRED data.

    Returns:
        (df, imputed_count)
    """
    cutoff_date = pd.Timestamp("2009-01-01")

    # 1. Identify rows that need imputation
    # Condition: Missing OR > 12 months after data date
    one_year_later = df["date"] + DateOffset(years=1)

    # Only consider imputation for dates before 2009-01-01
    needs_imputation = ((df["release_date"].isna()) | (df["release_date"] > one_year_later)) & (df["date"] < cutoff_date)

    if not needs_imputation.any():
        return df, 0

    # Prepare Candidate Dates

    # Candidate A: First Friday of NEXT YEAR'S February
    # (e.g., Data Jan 2020 -> Feb 2021)
    # Constructing as Series to preserve index
    next_year_feb_str = (df["date"] + DateOffset(years=1)).dt.year.astype(str) + "-02-01"
    next_year_feb = pd.to_datetime(next_year_feb_str)
    cand_feb_next_year = get_first_friday_of_month(next_year_feb)

    # Candidate B1: First Friday of M+1
    cand_m1 = get_first_friday_of_month(df["date"] + DateOffset(months=1))
    # Candidate B2: First Friday of M+2
    cand_m2 = get_first_friday_of_month(df["date"] + DateOffset(months=2))
    # Candidate B3: First Friday of M+3
    cand_m3 = get_first_friday_of_month(df["date"] + DateOffset(months=3))

    # We need to apply logic row by row or via masks. Using masks for speed.
    # Because snapshot_date is constant for the whole file, this is efficient.

    # --- LOGIC GATES ---

    # Gate 1: Snapshot is > 1 year after data date
    # OR
    # Gate 2: Snapshot includes next Feb (Snapshot >= cand_feb_next_year)
    # ACTION: Use cand_feb_next_year
    gate_1_2 = (snapshot_date > one_year_later) | (snapshot_date >= cand_feb_next_year)

    # Gate 3: Snapshot is closer (Option 3 logic)
    # Take latest of m1, m2, m3 AS LONG AS it is <= snapshot_date
    gate_3 = ~gate_1_2

    # Apply Option 1 & 2
    mask_1_2 = needs_imputation & gate_1_2
    df.loc[mask_1_2, "release_date"] = cand_feb_next_year[mask_1_2]

    # Apply Option 3
    # We need to find Max(m1, m2, m3) where date <= snapshot_date
    if gate_3.any():
        mask_3 = needs_imputation & gate_3

        # Create a mini dataframe to do the row-wise max logic efficiently
        candidates = pd.DataFrame({
            "m1": cand_m1[mask_3],
            "m2": cand_m2[mask_3],
            "m3": cand_m3[mask_3]
        }, index=df.index[mask_3]) # Ensure index alignment

        # Filter: Set dates > snapshot_date to NaT
        candidates[candidates > snapshot_date] = pd.NaT

        # Take the Max (latest valid date)
        best_date = candidates.max(axis=1)

        # Assign
        df.loc[mask_3, "release_date"] = best_date

    df["release_date"] = pd.to_datetime(df["release_date"])
    return df, int(needs_imputation.sum())

# ----------------------------------------
# FUTURE DATE EXTENSION HELPERS
# ----------------------------------------

def extend_target_with_complete_date_range(
    target_df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    end_date_setting: pd.Timestamp,
    is_univariate: bool = True,
    is_last_release: bool = False
) -> pd.DataFrame:
    """
    Ensure target file has a complete date range from window_start to END_DATE month.

    This function:
    1. Identifies all missing months in the range
    2. Scrapes BLS website for future release dates (after last existing data)
    3. Uses first Friday fallback for any missing dates
    4. Adds rows with NaN values for missing months
    5. Removes any future rows that are now past END_DATE (if it was reduced)

    Args:
        target_df: Existing target data with columns [ds, y, release_date] for univariate
                   or [ds, release_date, leaf_col_1, ...] for multivariate
        window_start: First month that should exist in target
        window_end: Last month with actual published data
        end_date_setting: END_DATE from settings (determines final month to include)
        is_univariate: True for single column (y), False for multivariate
        is_last_release: If True, future release dates are 3 months after observation.
                        If False, future release dates are 1 month after observation (first Friday).

    Returns:
        Extended DataFrame with complete date range
    """
    # Determine the final month we want in the target (based on END_DATE setting)
    final_month = pd.Timestamp(end_date_setting).to_period('M').to_timestamp()

    # Get leaf column names for multivariate case
    leaf_cols = []
    if not is_univariate and not target_df.empty:
        leaf_cols = [c for c in target_df.columns if c not in ['ds', 'release_date']]
        logger.debug(f"Multivariate extension: {len(leaf_cols)} leaf columns")

    # Generate complete range of months we should have
    all_months = pd.date_range(
        start=window_start,
        end=final_month,
        freq='MS'  # Month start
    )

    if target_df.empty:
        existing_months = set()
        last_data_month = window_start
    else:
        existing_months = set(target_df['ds'].dt.to_period('M'))
        last_data_month = target_df['ds'].max()

    # Identify missing months
    missing_months = [m for m in all_months if m.to_period('M') not in existing_months]

    if not missing_months:
        logger.debug("No missing months found in target date range")
        # Still need to check for future rows to remove
        if not target_df.empty:
            target_df = target_df[target_df['ds'] <= final_month].copy()
        return target_df

    logger.debug(f"Found {len(missing_months)} missing months in target range")

    # Separate missing months into: past/present (backfill) vs future (scrape)
    past_missing = [m for m in missing_months if m <= last_data_month]
    future_missing = [m for m in missing_months if m > last_data_month]

    new_rows = []

    def _create_row(month: pd.Timestamp, release_date: pd.Timestamp) -> dict:
        """Create a new row with appropriate columns for univariate or multivariate."""
        if is_univariate:
            return {
                'ds': month,
                'y': float('nan'),
                'release_date': release_date
            }
        else:
            row = {'ds': month, 'release_date': release_date}
            for col in leaf_cols:
                row[col] = float('nan')
            return row

    # Handle past missing months (backfill with first Friday)
    if past_missing:
        logger.debug(f"Backfilling {len(past_missing)} past missing months with first Friday rule")
        for month in past_missing:
            next_month = month + pd.DateOffset(months=1)
            release_date = get_first_friday_of_month(next_month)

            row = _create_row(month, release_date)
            new_rows.append(row)
            logger.warning(
                f"Backfilled missing month {month.strftime('%Y-%m')} "
                f"with first Friday: {release_date.strftime('%Y-%m-%d')}"
            )

    # Handle future missing months (scrape BLS)
    if future_missing:
        release_type = "last_release (3-month offset)" if is_last_release else "first_release"
        logger.debug(f"Scraping BLS for {len(future_missing)} future months ({release_type})")

        if is_last_release:
            # For last_release files: release dates are 3 months after observation
            # Example: Dec 2025 observation -> March 2026 release date
            # Scrape dates 3 months further out than the observation months
            future_start_for_scrape = future_missing[0] + pd.DateOffset(months=2)
            future_end_for_scrape = future_missing[-1] + pd.DateOffset(months=2)
            scraped_df = get_future_nfp_dates(future_start_for_scrape, future_end_for_scrape)

            # Build lookup dict from scraped data: release_month -> release_date
            scraped_lookup = {}
            for _, row in scraped_df.iterrows():
                release_month = row['observation_month'] + pd.DateOffset(months=1)
                scraped_lookup[release_month.to_period('M')] = row['release_date']

            # Map each observation month to its 3-month-out release date
            for obs_month in future_missing:
                release_month = obs_month + pd.DateOffset(months=3)
                release_period = release_month.to_period('M')

                if release_period in scraped_lookup:
                    release_date = scraped_lookup[release_period]
                else:
                    # Fallback: first Friday of release month
                    release_date = get_first_friday_of_month(release_month)
                    logger.warning(
                        f"No scraped date for {obs_month.strftime('%Y-%m')} last_release, "
                        f"using first Friday of {release_month.strftime('%Y-%m')}: {release_date}"
                    )

                new_row = _create_row(obs_month, release_date)
                new_rows.append(new_row)
        else:
            # For first_release files: release dates are 1 month after observation
            # Scrape BLS for future dates
            future_start = future_missing[0]
            future_end = future_missing[-1]

            scraped_df = get_future_nfp_dates(future_start, future_end)

            # Convert to target format
            for _, scraped_row in scraped_df.iterrows():
                new_row = _create_row(scraped_row['observation_month'], scraped_row['release_date'])
                new_rows.append(new_row)

    # Combine with existing data
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        result_df = pd.concat([target_df, new_df], ignore_index=True)
        result_df = result_df.sort_values('ds').reset_index(drop=True)
        logger.debug(f"Added {len(new_rows)} new rows to target")
    else:
        result_df = target_df.copy()

    # Remove any rows beyond final_month (in case END_DATE was reduced)
    rows_before = len(result_df)
    result_df = result_df[result_df['ds'] <= final_month].copy()
    rows_removed = rows_before - len(result_df)

    if rows_removed > 0:
        logger.debug(f"Removed {rows_removed} future rows that are now past END_DATE")

    return result_df

# ----------------------------------------

def save_report(report: Dict[str, Any]) -> None:
    _ensure_dir(FRED_ROOT)
    report["last_loaded"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    
    ordered_report = {
        "last_loaded": report.get("last_loaded"),
        "timestamp": report.get("timestamp"),
        "status": report.get("status"),
        "missing_configurations": report.get("missing_configurations"),
        "successful_series_count": report.get("successful_series_count"),
        "missing_series_count": report.get("missing_series_count"),
        "download_failures": report.get("download_failures"),
        "downloaded_keys": report.get("downloaded_keys"),
        "merged_series_count": report.get("merged_series_count")
    }
    final_report = {k: v for k, v in ordered_report.items() if v is not None}

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, default=str)
    logger.debug(f"Saved consolidated report to {REPORT_FILE}")

def month_ends(start_date: str, end_date: str) -> List[pd.Timestamp]:
    s = pd.to_datetime(start_date).to_period("M").to_timestamp("M")
    e = pd.to_datetime(end_date).to_period("M").to_timestamp("M")
    return list(pd.date_range(s, e, freq="M"))

def _fetch_one_all_asof(fred: Fred, fred_id: str, uid: str, as_of_str: str) -> pd.DataFrame:
    """
    Fetch FRED series with vintage history and fix data quality issues.

    Handles:
    1. Negative lags (release before observation - impossible)
    2. Extreme lags (>10 years - likely data errors)
    3. Missing release dates

    Does NOT create synthetic vintages for legitimate delays (e.g., COVID, government shutdowns).
    Only replaces problematic vintages, never creates duplicates.
    """
    df = fred.get_series_as_of_date(fred_id, as_of_date=as_of_str)
    if df is None or df.empty:
        return pd.DataFrame(columns=["unique_id","ds","y","realtime_start"])

    df = df.rename(columns={"date": "ds", "value": "y"})
    df["unique_id"] = uid
    df["ds"] = pd.to_datetime(df["ds"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").astype("float32")

    # Calculate reliable typical_lag from recent clean data
    as_of_dt = pd.to_datetime(as_of_str)

    # Use 5 years of recent data for lag calculation (more stable than 2 years)
    recent_cutoff = as_of_dt - pd.DateOffset(years=5)
    recent_df = df[df['ds'] >= recent_cutoff].copy()

    typical_lag = pd.Timedelta(days=45)  # Default for monthly employment data

    if len(recent_df) >= 24:  # Need at least 2 years of data
        recent_first_vintages = recent_df.groupby('ds')['realtime_start'].min()
        if not recent_first_vintages.empty:
            recent_lags = recent_first_vintages - recent_first_vintages.index

            # Filter outliers using IQR method for robustness
            q25 = recent_lags.quantile(0.25)
            q75 = recent_lags.quantile(0.75)
            iqr = q75 - q25

            # Only use lags within 1.5*IQR of the quartiles
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            filtered_lags = recent_lags[
                (recent_lags >= lower_bound) &
                (recent_lags <= upper_bound)
            ]

            if len(filtered_lags) >= 12:  # Need at least 1 year of clean data
                median_lag = filtered_lags.median()
                if not pd.isna(median_lag):
                    typical_lag = median_lag

    # Identify data quality issues (not legitimate delays)
    earliest_vintage = df.groupby('ds')['realtime_start'].min()
    actual_lags = earliest_vintage - earliest_vintage.index

    cutoff_date = pd.Timestamp("2009-01-01")

    # Define data quality issue criteria
    data_quality_issues_mask = (
        # Case 1: Negative lag (impossible - release before observation)
        (actual_lags < pd.Timedelta(days=0))
        |
        # Case 2: Extreme lag (>10 years - likely data error, not real delay)
        (actual_lags > pd.Timedelta(days=3650))
        |
        # Case 3: Missing release dates (NaT)
        (earliest_vintage.isna())
    )

    # Only fix issues for dates before 2009-01-01
    data_quality_issues_mask = data_quality_issues_mask & (earliest_vintage.index < cutoff_date)

    if data_quality_issues_mask.any():
        issue_dates = earliest_vintage[data_quality_issues_mask].index

        # Strategy: Replace ONLY the problematic first vintage, keep all other vintages
        # For each issue_date, we'll:
        # 1. Remove the earliest (problematic) vintage
        # 2. Add a corrected first vintage with typical_lag

        # Identify the problematic vintages (earliest per observation date)
        problematic_vintages = []
        for date in issue_dates:
            date_vintages = df[df['ds'] == date].sort_values('realtime_start')
            if not date_vintages.empty:
                # The first vintage is the problematic one
                problematic_vintages.append(date_vintages.index[0])

        # Remove problematic vintages
        df_clean = df.drop(problematic_vintages).copy()

        # Create corrected first vintages using First Friday rule
        corrected_rows = []
        for date in issue_dates:
            # Get any row for this observation date (for the value)
            sample_row = df[df['ds'] == date].iloc[0].copy()

            # Calculate First Friday of next month
            next_month = date + pd.DateOffset(months=1)
            first_of_next = next_month.replace(day=1)
            # Days to add to reach Friday (weekday 4)
            days_to_friday = (4 - first_of_next.weekday() + 7) % 7
            first_friday = first_of_next + pd.Timedelta(days=days_to_friday)

            # Set corrected release date to First Friday
            sample_row['realtime_start'] = first_friday
            corrected_rows.append(sample_row)

        if corrected_rows:
            corrected_df = pd.DataFrame(corrected_rows)
            # Combine clean data with corrected vintages
            df = pd.concat([df_clean, corrected_df], ignore_index=True)

        logger.debug(
            f"{uid}: Fixed {len(issue_dates)} observations with data quality issues "
            f"(using First Friday of next month rule) - only for dates before 2009-01-01"
        )

    return df[["unique_id","ds","y","realtime_start"]]

# Thread-safe rate limiting for parallel requests
_request_lock = threading.Lock()
_last_request_time = [0.0]

def _fetch_with_retry(fred, fid, uid, as_of_str, max_retries=3, base_delay=30):
    """Fetch with exponential backoff on rate limit errors."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return _fetch_one_all_asof(fred, fid, uid, as_of_str)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            is_rate_limit = "rate limit" in error_str or "too many requests" in error_str
            is_parse_error = "mismatched tag" in error_str or "xml" in error_str

            if (is_rate_limit or is_parse_error) and attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)  # 30s, 60s, 120s
                logger.warning(f"Retrying {uid} in {wait_time}s (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(wait_time)
                continue
            break
    raise last_error

def _rate_limited_fetch(fred, fid, uid, as_of_str, per_request_delay, max_retries):
    """Wrapper enforcing minimum delay between requests across all threads."""
    with _request_lock:
        elapsed = time.time() - _last_request_time[0]
        if elapsed < per_request_delay:
            time.sleep(per_request_delay - elapsed)
        _last_request_time[0] = time.time()
    return _fetch_with_retry(fred, fid, uid, as_of_str, max_retries)

def download_master_audit(
    end_date: str = END_DATE,
    codes: Dict[str, Optional[str]] = FRED_EMPLOYMENT_CODES,
    batch_size: int = 80,
    pause_seconds: int = 45,        # Reduced from 60 (retries handle rate limits)
    use_disk_cache: bool = True,
    specific_codes: Optional[List[str]] = None,
    max_workers: int = 3,           # Limited parallelization
    per_request_delay: float = 0.8, # Delay between requests
    max_retries: int = 3,           # Retry failed requests
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    _ensure_dir(FRED_ROOT)
    as_of_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    audit_file = FRED_ROOT / f"_audit_asof_{as_of_str}.parquet"

    report = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "download_failures": {},
        "successful_series_count": 0,
        "downloaded_keys": [],
    }

    report["missing_configurations"] = [k for k, v in codes.items() if v is None]

    if use_disk_cache and audit_file.exists() and not REFRESH_CACHE and specific_codes is None:
        logger.debug(f"Loading audit cache → {audit_file.name}")
        audit = pd.read_parquet(audit_file, engine="pyarrow")
        audit["unique_id"] = audit["unique_id"].astype("category")
        audit = _align_audit_to_total_calendar(audit, total_uid="total_nsa")
        report["successful_series_count"] = len(audit["unique_id"].unique())
        return audit, report

    logger.info("Download starting.")

    if specific_codes is not None:
        items = [(u, codes[u]) for u in specific_codes if codes.get(u)]
        download_groups = [items]
    else:
        total_items = [(u, codes[u]) for u in TOTAL_UIDS if codes.get(u)]
        other_items = [(u, f) for u, f in codes.items() if f and u not in TOTAL_UIDS]
        download_groups = [total_items, other_items]
        items = total_items + other_items

    if not items:
        return pd.DataFrame(columns=["unique_id","ds","y","realtime_start"]), report

    frames = []
    fred = Fred(api_key=FRED_API_KEY)

    def _download_items(items_to_fetch: List[Tuple[str, str]]) -> None:
        if not items_to_fetch:
            return
        n_batches = math.ceil(len(items_to_fetch) / batch_size)
        for b in range(n_batches):
            batch_items = items_to_fetch[b*batch_size : (b+1)*batch_size]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_uid = {
                    executor.submit(
                        _rate_limited_fetch, fred, fid, uid, as_of_str,
                        per_request_delay, max_retries
                    ): uid
                    for uid, fid in batch_items
                }

                for future in as_completed(future_to_uid):
                    uid = future_to_uid[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            frames.append(df)
                            report["downloaded_keys"].append(uid)
                    except Exception as e:
                        logger.warning(f"  Failed after retries: {uid} @ {as_of_str}: {e}")
                        report["download_failures"][uid] = str(e)

            if b < n_batches - 1:
                time.sleep(pause_seconds)

    if specific_codes is None:
        logger.info("Phase 1 starting.")
        _download_items(download_groups[0])
        logger.info("Phase 1 complete.")
        if len(download_groups) > 1 and download_groups[1]:
            logger.info("Starting phase 2.")
            _download_items(download_groups[1])
    else:
        for group in download_groups:
            _download_items(group)

    report["successful_series_count"] = len(report["downloaded_keys"])
    report["missing_series_count"] = len(report["download_failures"])

    if not frames:
        if specific_codes: return pd.DataFrame(columns=["unique_id","ds","y","realtime_start"]), report
        raise RuntimeError("Download produced no data.")

    new_data = pd.concat(frames, ignore_index=True)
    new_data["unique_id"] = new_data["unique_id"].astype("category")
    new_data = _align_audit_to_total_calendar(new_data, total_uid="total_nsa")
    
    if specific_codes is None and use_disk_cache:
         new_data = new_data.sort_values(["unique_id", "ds", "realtime_start"])
         _write_parquet(new_data, audit_file, dict_cols=["unique_id"])
         logger.debug(f"Wrote audit cache → {audit_file}")

    return new_data, report

def collapse_latest_asof(audit: pd.DataFrame, as_of_cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Revised collapse function:
    1. Imputes total_nsa first-release dates BEFORE snapshot filtering.
    2. Propagates total_nsa release calendar to all series.
    3. Uses strict < instead of <= to prevent same-day data leakage.
    """
    if audit.empty:
        return pd.DataFrame(columns=["date", "value", "series_name", "series_code", "release_date"])

    aligned = audit.copy()
    imputed_count = 0

    total_uid = "total_nsa" if "total_nsa" in aligned["unique_id"].values else None
    if total_uid:
        total = aligned[aligned["unique_id"] == total_uid].copy()
        total = total.sort_values(["ds", "realtime_start"])
        total["vintage_idx"] = total.groupby("ds").cumcount()

        first_release = total[total["vintage_idx"] == 0][["ds", "realtime_start"]].rename(
            columns={"ds": "date", "realtime_start": "release_date"}
        )
        first_release, imputed_count = calculate_complex_release_date(first_release, as_of_cutoff)
        first_map = first_release.set_index("date")["release_date"]

        mask_first = total["vintage_idx"] == 0
        total.loc[mask_first, "realtime_start"] = total.loc[mask_first, "ds"].map(first_map)
        total["realtime_start"] = pd.to_datetime(total["realtime_start"])

        calendar = total[["ds", "vintage_idx", "realtime_start"]].rename(
            columns={"realtime_start": "total_realtime_start"}
        )

        aligned = aligned.sort_values(["unique_id", "ds", "realtime_start"]).copy()
        aligned["vintage_idx"] = aligned.groupby(["unique_id", "ds"]).cumcount()
        aligned = aligned.merge(calendar, on=["ds", "vintage_idx"], how="left")
        aligned["realtime_start"] = aligned["total_realtime_start"].combine_first(aligned["realtime_start"])
        aligned = aligned.drop(columns=["vintage_idx", "total_realtime_start"])

    # Changed from <= to strict < to prevent same-day data leakage
    df = aligned[aligned["realtime_start"] < as_of_cutoff]
    if df.empty:
        return pd.DataFrame(columns=["date", "value", "series_name", "series_code", "release_date"])

    out = (
        df.drop_duplicates(["unique_id", "ds"], keep="last")
          .loc[:, ["unique_id", "ds", "y", "realtime_start"]]
          .reset_index(drop=True)
    )

    out = out.rename(columns={"ds": "date", "y": "value", "realtime_start": "release_date"})
    out["series_name"] = out["unique_id"]
    out["series_code"] = out["unique_id"].map(FRED_EMPLOYMENT_CODES)

    if not total_uid:
        out, imputed_count = calculate_complex_release_date(out, as_of_cutoff)

    out["release_date"] = pd.to_datetime(out["release_date"])
    logger.info(f"Snapshot {as_of_cutoff.date()}: imputed {imputed_count} dates")

    out = out[["date", "value", "series_name", "series_code", "release_date"]]

    out["series_name"] = out["series_name"].astype("category")
    out["series_code"] = out["series_code"].astype("category")
    return out

def _get_wide_leaf_nodes_with_dates(audit, leaf_uids, mode='last', require_complete=False):
    """
    Convert long-format audit data into wide-format with leaf node series as columns.

    Args:
        audit: DataFrame with columns [unique_id, ds, y, realtime_start]
        leaf_uids: List of leaf node unique_ids to include
        mode: 'first' for first release, 'last' for latest release
        require_complete: If True, filter to only rows where ALL leaf columns have data

    Returns:
        DataFrame with columns [ds, release_date, leaf_col_1, leaf_col_2, ...]
    """
    df = audit[audit["unique_id"].isin(leaf_uids)].copy()
    if df.empty: return pd.DataFrame(columns=["ds"] + leaf_uids + ["release_date"])

    df = df.sort_values(["unique_id", "ds", "realtime_start"])
    keep = 'first' if mode == 'first' else 'last'
    df = df.drop_duplicates(["unique_id", "ds"], keep=keep)

    wide = df.pivot(index="ds", columns="unique_id", values="y").reset_index()
    for u in leaf_uids:
        if u not in wide.columns: wide[u] = float('nan')

    # Filter to complete rows only (all leaf columns have data)
    if require_complete:
        leaf_cols = [u for u in leaf_uids if u in wide.columns]
        rows_before = len(wide)
        complete_mask = wide[leaf_cols].notna().all(axis=1)
        wide = wide[complete_mask].copy()
        rows_after = len(wide)

        if rows_after < rows_before:
            logger.debug(f"Complete rows filter: {rows_before} -> {rows_after} rows")
            if rows_after > 0:
                logger.debug(f"Effective date range: {wide['ds'].min()} to {wide['ds'].max()}")

    date_source_uid = "total_nsa" if "total_nsa" in audit["unique_id"].values else leaf_uids[0]

    date_df = audit[audit["unique_id"] == date_source_uid].copy()
    if not date_df.empty:
        date_df = date_df.drop_duplicates(["ds"], keep=keep)[["ds", "realtime_start"]]
        date_df = date_df.rename(columns={"realtime_start": "release_date"})
        wide = pd.merge(wide, date_df, on="ds", how="left")
    else:
        max_dates = df.groupby("ds")["realtime_start"].max().reset_index()
        max_dates = max_dates.rename(columns={"realtime_start": "release_date"})
        wide = pd.merge(wide, max_dates, on="ds", how="left")

    cols = ["ds", "release_date"] + sorted(leaf_uids)
    return wide[cols]

def _get_single_series(audit, uid, mode='first'):
    df = audit[audit["unique_id"] == uid]
    if df.empty: return pd.DataFrame(columns=["ds","y", "release_date"])

    # Explicit sort to ensure correct first/last release selection
    df = df.sort_values(["ds", "realtime_start"])
    keep = 'first' if mode == 'first' else 'last'
    df = df.drop_duplicates(["ds"], keep=keep).loc[:, ["ds","y", "realtime_start"]].reset_index(drop=True)
    df = df.rename(columns={"realtime_start": "release_date"})
    return df

def _slice_to_window(df, start, end):
    if df.empty: return df
    return df[(df["ds"] >= start) & (df["ds"] <= end)].reset_index(drop=True)

def _process_and_save_targets(targets, paths, window_start, window_end,
                               end_date_setting, snapshot_date, is_univariate):
    """
    Process and save target files with appropriate imputation and extension.

    All files get a final row for END_DATE month (e.g., Jan 2026) with y=NaN.

    For first_release files:
        - Historical data keeps y values as-is
        - Final month (Jan 2026) has y=NaN, release_date = first Friday of next month (Feb 2026)

    For last_release files:
        - Last 3 months (Nov, Dec 2025 + Jan 2026) have y=NaN
        - Release dates for these 3 months = scraped date 3 months after observation

    Args:
        targets: Dict of {key: DataFrame} with target data
        paths: Dict of {key: Path} for output files
        window_start: Start of data window
        window_end: End of data window
        end_date_setting: END_DATE from settings
        snapshot_date: Date when audit was downloaded
        is_univariate: If True, processing univariate files. If False, multivariate (apply dropna).
    """
    final_month = pd.Timestamp(end_date_setting).to_period('M').to_timestamp()

    for key, df in targets.items():
        # DISABLED: Skip last release targets - focusing on first release only
        if "last" in key:
            logger.debug(f"Skipping {key} (last release disabled)")
            continue

        df_sliced = _slice_to_window(df, window_start, window_end)

        if "first" in key:
            # SIMPLE logic for "first release" files (14 day threshold)
            df_imputed = impute_target_release_date_simple(df_sliced)

            # For multivariate: dropna to ensure full matrix
            if not is_univariate:
                leaf_cols = [c for c in df_imputed.columns if c not in ['ds', 'release_date']]
                rows_before = len(df_imputed)
                df_imputed = df_imputed.dropna(subset=leaf_cols).reset_index(drop=True)
                logger.debug(f"Multivariate {key}: dropped {rows_before - len(df_imputed)} incomplete rows")

            # Get the last month with actual data
            last_data_month = df_imputed['ds'].max() if not df_imputed.empty else window_start

            # Identify all months that need placeholder rows (unreleased data)
            # This includes any months between last_data_month and final_month (inclusive)
            months_to_add = []
            current_month = last_data_month + pd.DateOffset(months=1)
            current_month = current_month.to_period('M').to_timestamp()  # Normalize to month start

            while current_month <= final_month:
                months_to_add.append(current_month)
                current_month = current_month + pd.DateOffset(months=1)

            logger.debug(f"Months needing placeholders: {[m.strftime('%Y-%m') for m in months_to_add]}")

            # Keep only data up to last_data_month (trim any partial/future data)
            df_final = df_imputed[df_imputed['ds'] <= last_data_month].copy()

            # Get release dates for all placeholder months
            if months_to_add:
                scraped_df = get_future_nfp_dates(months_to_add[0], months_to_add[-1])
                scraped_lookup = {row['observation_month']: row['release_date']
                                  for _, row in scraped_df.iterrows()}

                # Add placeholder rows for each unreleased month
                for month in months_to_add:
                    if month in scraped_lookup:
                        release_date = scraped_lookup[month]
                    else:
                        release_date = get_first_friday_of_month(month + pd.DateOffset(months=1))
                        logger.warning(f"No scraped date for {month.strftime('%Y-%m')}, using first Friday: {release_date}")

                    if is_univariate:
                        new_row = pd.DataFrame({
                            'ds': [month],
                            'y': [float('nan')],
                            'release_date': [release_date]
                        })
                    else:
                        leaf_cols = [c for c in df_final.columns if c not in ['ds', 'release_date']]
                        new_row = pd.DataFrame({'ds': [month], 'release_date': [release_date]})
                        for col in leaf_cols:
                            new_row[col] = float('nan')

                    df_final = pd.concat([df_final, new_row], ignore_index=True)
                    logger.debug(f"Added {month.strftime('%Y-%m')} with NaN, release_date: {release_date}")

            _write_parquet(df_final, paths[key])
            logger.debug(f"Saved {key} with {len(df_final)} rows (ending at {final_month.strftime('%Y-%m')})")

        else:  # "last" in key
            # COMPLEX logic for "last release" files
            df_imputed = impute_target_release_date_complex(df_sliced, snapshot_date)

            # For multivariate: dropna to ensure full matrix
            if not is_univariate:
                leaf_cols = [c for c in df_imputed.columns if c not in ['ds', 'release_date']]
                rows_before = len(df_imputed)
                df_imputed = df_imputed.dropna(subset=leaf_cols).reset_index(drop=True)
                logger.debug(f"Multivariate {key}: dropped {rows_before - len(df_imputed)} incomplete rows")

            # Trim to before final_month
            df_final = df_imputed[df_imputed['ds'] < final_month].copy()

            # Add final_month row with NaN values (placeholder release date, will update below)
            if is_univariate:
                final_row = pd.DataFrame({
                    'ds': [final_month],
                    'y': [float('nan')],
                    'release_date': [pd.NaT]
                })
            else:
                leaf_cols = [c for c in df_final.columns if c not in ['ds', 'release_date']]
                final_row = pd.DataFrame({'ds': [final_month], 'release_date': [pd.NaT]})
                for col in leaf_cols:
                    final_row[col] = float('nan')

            df_final = pd.concat([df_final, final_row], ignore_index=True)

            # For last_release: last 3 months (including final_month) get y=NaN and release_date = 3 months out
            # Get last 3 months
            last_3_months = df_final['ds'].nlargest(3).values
            last_3_mask = df_final['ds'].isin(last_3_months)

            logger.debug(f"Last_release {key}: Setting NaN for last 3 months: {[pd.Timestamp(m).strftime('%Y-%m') for m in sorted(last_3_months)]}")

            # Set y values to NaN for last 3 months
            if is_univariate:
                df_final.loc[last_3_mask, 'y'] = float('nan')
            else:
                value_cols = [c for c in df_final.columns if c not in ['ds', 'release_date']]
                for col in value_cols:
                    df_final.loc[last_3_mask, col] = float('nan')

            # Scrape release dates for the last 3 months (3 months out)
            obs_months = sorted([pd.Timestamp(m) for m in last_3_months])

            # Scrape dates for the release months (observation + 3 months)
            release_start = obs_months[0] + pd.DateOffset(months=2)
            release_end = obs_months[-1] + pd.DateOffset(months=2)

            scraped_df = get_future_nfp_dates(release_start, release_end)

            # Build lookup: release_month -> release_date
            scraped_lookup = {}
            for _, row in scraped_df.iterrows():
                release_month = row['observation_month'] + pd.DateOffset(months=1)
                scraped_lookup[release_month.to_period('M')] = row['release_date']

            # Update release dates for last 3 months
            for obs_month in obs_months:
                release_month = obs_month + pd.DateOffset(months=3)
                release_period = release_month.to_period('M')

                if release_period in scraped_lookup:
                    new_release_date = scraped_lookup[release_period]
                else:
                    new_release_date = get_first_friday_of_month(release_month)
                    logger.warning(f"No scraped date for {obs_month.strftime('%Y-%m')} last_release, using first Friday: {new_release_date}")

                df_final.loc[df_final['ds'] == obs_month, 'release_date'] = new_release_date
                logger.debug(f"  {obs_month.strftime('%Y-%m')} -> release_date: {new_release_date}")

            _write_parquet(df_final, paths[key])
            logger.debug(f"Saved {key} with {len(df_final)} rows (ending at {final_month.strftime('%Y-%m')})")

def build_nfp_target_files(audit, window_start, window_end, snapshot_date):
    """
    Build NFP target files for both univariate and multivariate models.

    Generates up to 8 files:
    - 4 univariate files (total NFP only): total_nsa_first_release, total_nsa_last_release,
      total_sa_first_release, total_sa_last_release
    - 4 multivariate files (leaf nodes, full matrix): y_nsa_first_release, y_nsa_last_release,
      y_sa_first_release, y_sa_last_release

    When MODEL_TYPE = "multivariate": generates all 8 files
    When MODEL_TYPE = "univariate": generates only 4 univariate files
    """
    _ensure_dir(NFP_target_DIR)

    is_multivariate = (MODEL_TYPE != "univariate")
    series_names = list(FRED_EMPLOYMENT_CODES.keys())
    end_date_setting = pd.to_datetime(END_DATE)

    # ========== ALWAYS BUILD UNIVARIATE (Total NFP) ==========
    # NOTE: Last release targets disabled - focusing on first release only
    logger.debug("Building UNIVARIATE targets (Total NFP) - First Release Only.")
    total_nsa_first = _get_single_series(audit, "total_nsa", mode='first')
    # total_nsa_last = _get_single_series(audit, "total_nsa", mode='last')  # DISABLED
    total_sa_first = _get_single_series(audit, "total", mode='first')
    # total_sa_last = _get_single_series(audit, "total", mode='last')  # DISABLED

    univariate_targets = {
        "total_nsa_first": total_nsa_first,
        # "total_nsa_last": total_nsa_last,  # DISABLED
        "total_sa_first": total_sa_first,
        # "total_sa_last": total_sa_last,  # DISABLED
    }

    univariate_paths = {
        "total_nsa_first": NFP_target_DIR / "total_nsa_first_release.parquet",
        # "total_nsa_last": NFP_target_DIR / "total_nsa_last_release.parquet",  # DISABLED
        "total_sa_first": NFP_target_DIR / "total_sa_first_release.parquet",
        # "total_sa_last": NFP_target_DIR / "total_sa_last_release.parquet",  # DISABLED
    }

    # ========== BUILD MULTIVARIATE (Leaf Nodes) IF ENABLED ==========
    # NOTE: Last release targets disabled - focusing on first release only
    if is_multivariate:
        logger.debug("Building MULTIVARIATE targets (Leaf Nodes) - First Release Only.")

        # NSA leaf nodes
        _, _, nsa_bottom_uids = build_hierarchy_structure(series_names, include_nsa=True)
        logger.debug(f"NSA leaf nodes: {len(nsa_bottom_uids)} series")

        # SA leaf nodes
        _, _, sa_bottom_uids = build_hierarchy_structure(series_names, include_nsa=False)
        logger.debug(f"SA leaf nodes: {len(sa_bottom_uids)} series")

        # Get wide-format data (do NOT apply require_complete here - dropna happens in _process_and_save_targets)
        y_nsa_first = _get_wide_leaf_nodes_with_dates(audit, nsa_bottom_uids, 'first', require_complete=False)
        # y_nsa_last = _get_wide_leaf_nodes_with_dates(audit, nsa_bottom_uids, 'last', require_complete=False)  # DISABLED
        y_sa_first = _get_wide_leaf_nodes_with_dates(audit, sa_bottom_uids, 'first', require_complete=False)
        # y_sa_last = _get_wide_leaf_nodes_with_dates(audit, sa_bottom_uids, 'last', require_complete=False)  # DISABLED

        multivariate_targets = {
            "y_nsa_first": y_nsa_first,
            # "y_nsa_last": y_nsa_last,  # DISABLED
            "y_sa_first": y_sa_first,
            # "y_sa_last": y_sa_last,  # DISABLED
        }

        multivariate_paths = {
            "y_nsa_first": NFP_target_DIR / "y_nsa_first_release.parquet",
            # "y_nsa_last": NFP_target_DIR / "y_nsa_last_release.parquet",  # DISABLED
            "y_sa_first": NFP_target_DIR / "y_sa_first_release.parquet",
            # "y_sa_last": NFP_target_DIR / "y_sa_last_release.parquet",  # DISABLED
        }
    else:
        multivariate_targets = {}
        multivariate_paths = {}

    # ========== PROCESS AND SAVE ALL FILES ==========

    # Process univariate files (always is_univariate=True)
    logger.debug("Processing and saving UNIVARIATE files...")
    _process_and_save_targets(
        univariate_targets, univariate_paths,
        window_start, window_end, end_date_setting, snapshot_date,
        is_univariate=True
    )

    # Process multivariate files (is_univariate=False, apply dropna)
    if multivariate_targets:
        logger.debug("Processing and saving MULTIVARIATE files...")
        _process_and_save_targets(
            multivariate_targets, multivariate_paths,
            window_start, window_end, end_date_setting, snapshot_date,
            is_univariate=False
        )

    return {**univariate_paths, **multivariate_paths}

def build_monthly_snapshots_from_audit(audit, start_date, end_date, refresh_existing, limit_months):
    sched = month_ends(start_date, end_date)
    if limit_months: sched = sched[:int(limit_months)]
    
    for i, as_of in enumerate(sched, 1):
        out = month_file_path(as_of)
        if out.exists() and not refresh_existing: 
            logger.debug(f"[{i}/{len(sched)}] {as_of.date()} exists.")
            continue
        
        # Pass median_lags removed; logic is now self-contained in calculate_complex_release_date
        collapsed = collapse_latest_asof(audit, as_of)
        
        if not collapsed.empty:
            _write_parquet(collapsed, out, dict_cols=["series_name", "series_code"])
            
    return sched[0], sched[-1]

def build_all_snapshots(start_date=START_DATE, end_date=END_DATE, refresh_existing=False):
    as_of = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    audit_file = FRED_ROOT / f"_audit_asof_{as_of}.parquet"

    # Skip-if-exists check: verify if all data already exists
    if not refresh_existing:
        sched = month_ends(start_date, end_date)
        all_snapshots_exist = all(month_file_path(m).exists() for m in sched)

        # Check NFP target files
        nfp_target_files = [
            NFP_target_DIR / "total_nsa_first_release.parquet",
            NFP_target_DIR / "total_sa_first_release.parquet",
        ]
        all_targets_exist = all(f.exists() for f in nfp_target_files)

        if audit_file.exists() and all_snapshots_exist and all_targets_exist:
            print(f"✓ FRED data already exists: {len(sched)} monthly snapshots", flush=True)
            print(f"  Date range: {sched[0].date()} to {sched[-1].date()}", flush=True)
            logger.info(f"All FRED data exists, skipping download (use --refresh to force)")
            return

    if refresh_existing or not audit_file.exists():
        audit, _ = download_master_audit(end_date=end_date)
    else:
        audit = pd.read_parquet(audit_file)

    s, e = build_monthly_snapshots_from_audit(audit, start_date, end_date, refresh_existing, None)

    # Use end_date as snapshot_date since audit was downloaded "as of" this date
    snapshot_date = pd.to_datetime(end_date)
    build_nfp_target_files(audit, s, e, snapshot_date)
    logger.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NFP Snapshots")
    parser.add_argument(
        "--refresh", 
        action="store_true", 
        help="If set, forces a refresh of existing snapshots."
    )
    
    args = parser.parse_args()
    build_all_snapshots(refresh_existing=args.refresh)
