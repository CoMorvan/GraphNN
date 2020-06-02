from openbabel import pybel

import chemopy.basak as basak
import chemopy.bcut as bcut
import chemopy.charge as charge
import chemopy.connectivity as connectivity
import chemopy.constitution as constitution
import chemopy.estate as estate
import chemopy.kappa as kappa
import chemopy.moe as moe
import chemopy.molproperty as molproperty
import chemopy.topology as topology
import numpy as np
from rdkit import Chem


def load_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)

    data_constitution = constitution.GetConstitutional(mol)
    data_topology = topology.GetTopology(mol)
    data_connectivity = connectivity.GetConnectivity(mol)
    data_kappa = kappa.GetKappa(mol)
    data_bcut = bcut.GetBurden(mol)
    data_estate = estate.GetEstate(mol)
    data_basak = basak.Getbasak(mol)
    data_property = molproperty.GetMolecularProperty(mol)
    data_charge = charge.GetCharge(mol)
    data_moe = moe.GetMOE(mol)

    data = dict(data_constitution)
    data.update(data_topology)
    data.update(data_connectivity)
    data.update(data_kappa)
    data.update(data_bcut)
    data.update(data_estate)
    data.update(data_basak)
    data.update(data_property)
    data.update(data_charge)
    data.update(data_moe)

    fingeprints = []
    fingeprints.append(data['Weight'])
    fingeprints.append(data['nhyd'])
    fingeprints.append(data['nhal'])
    fingeprints.append(data['nhet'])
    fingeprints.append(data['nhev'])
    fingeprints.append(data['ncof'])
    fingeprints.append(data['ncocl'])
    fingeprints.append(data['ncobr'])
    fingeprints.append(data['ncoi'])
    fingeprints.append(data['ncarb'])
    fingeprints.append(data['nphos'])
    fingeprints.append(data['nsulph'])
    fingeprints.append(data['noxy'])
    fingeprints.append(data['nnitro'])
    fingeprints.append(data['nring'])
    fingeprints.append(data['nrot'])
    fingeprints.append(data['ndonr'])
    fingeprints.append(data['naccr'])
    fingeprints.append(data['nsb'])
    fingeprints.append(data['ndb'])
    fingeprints.append(data['Tsch'])
    fingeprints.append(data['Tigdi'])
    fingeprints.append(data['Platt'])
    fingeprints.append(data['Xu'])
    fingeprints.append(data['Pol'])
    fingeprints.append(data['DZ'])
    fingeprints.append(data['Ipc'])
    fingeprints.append(data['BertzCT'])
    fingeprints.append(data['GMTI'])
    fingeprints.append(data['ZM1'])
    fingeprints.append(data['ZM2'])
    fingeprints.append(data['MZM1'])
    fingeprints.append(data['MZM2'])
    fingeprints.append(data['Qindex'])
    fingeprints.append(data['diametert'])
    fingeprints.append(data['radiust'])
    fingeprints.append(data['petitjeant'])
    fingeprints.append(data['Sito'])
    fingeprints.append(data['Hato'])
    fingeprints.append(data['Geto'])
    fingeprints.append(data['ISIZ'])
    fingeprints.append(data['TIAC'])
    fingeprints.append(data['ntb'])
    fingeprints.append(data['naro'])
    fingeprints.append(data['nta'])
    fingeprints.append(data['PC1'])
    fingeprints.append(data['PC2'])
    fingeprints.append(data['PC3'])
    fingeprints.append(data['PC4'])
    fingeprints.append(data['PC5'])
    fingeprints.append(data['PC6'])

    fingeprints.append(data['W'])
    fingeprints.append(data['AW'])
    fingeprints.append(data['J'])
    fingeprints.append(data['Thara'])

    fingeprints.append(data['IC0'])
    fingeprints.append(data['IC1'])
    fingeprints.append(data['IC2'])
    fingeprints.append(data['IC3'])
    fingeprints.append(data['IC4'])
    fingeprints.append(data['IC5'])
    fingeprints.append(data['IC6'])
    fingeprints.append(data['SIC0'])
    fingeprints.append(data['SIC1'])
    fingeprints.append(data['SIC2'])
    fingeprints.append(data['SIC3'])
    fingeprints.append(data['SIC4'])
    fingeprints.append(data['SIC5'])
    fingeprints.append(data['SIC6'])
    fingeprints.append(data['CIC0'])
    fingeprints.append(data['CIC1'])
    fingeprints.append(data['CIC2'])
    fingeprints.append(data['CIC3'])
    fingeprints.append(data['CIC4'])
    fingeprints.append(data['CIC5'])
    fingeprints.append(data['CIC6'])

    fingeprints.append(data['bcutp1'])
    fingeprints.append(data['bcutp2'])
    fingeprints.append(data['bcutp3'])
    fingeprints.append(data['bcutp4'])
    fingeprints.append(data['bcutp5'])
    fingeprints.append(data['bcutp6'])
    fingeprints.append(data['bcutp7'])
    fingeprints.append(data['bcutp8'])
    fingeprints.append(data['bcutp9'])
    fingeprints.append(data['bcutp10'])
    fingeprints.append(data['bcutp11'])
    fingeprints.append(data['bcutp12'])
    fingeprints.append(data['bcutp13'])
    fingeprints.append(data['bcutp14'])
    fingeprints.append(data['bcutp15'])
    fingeprints.append(data['bcutp16'])
    fingeprints.append(data['bcutm1'])
    fingeprints.append(data['bcutm2'])
    fingeprints.append(data['bcutm3'])
    fingeprints.append(data['bcutm4'])
    fingeprints.append(data['bcutm5'])
    fingeprints.append(data['bcutm6'])
    fingeprints.append(data['bcutm7'])
    fingeprints.append(data['bcutm8'])
    fingeprints.append(data['bcutm9'])
    fingeprints.append(data['bcutm10'])
    fingeprints.append(data['bcutm11'])
    fingeprints.append(data['bcutm12'])
    fingeprints.append(data['bcutm13'])
    fingeprints.append(data['bcutm14'])
    fingeprints.append(data['bcutm15'])
    fingeprints.append(data['bcutm16'])
    fingeprints.append(data['bcute1'])
    fingeprints.append(data['bcute2'])
    fingeprints.append(data['bcute3'])
    fingeprints.append(data['bcute4'])
    fingeprints.append(data['bcute5'])
    fingeprints.append(data['bcute6'])
    fingeprints.append(data['bcute7'])
    fingeprints.append(data['bcute8'])
    fingeprints.append(data['bcute9'])
    fingeprints.append(data['bcute10'])
    fingeprints.append(data['bcute11'])
    fingeprints.append(data['bcute12'])
    fingeprints.append(data['bcute13'])
    fingeprints.append(data['bcute14'])
    fingeprints.append(data['bcute15'])
    fingeprints.append(data['bcute16'])
    fingeprints.append(data['bcutv1'])
    fingeprints.append(data['bcutv2'])
    fingeprints.append(data['bcutv3'])
    fingeprints.append(data['bcutv4'])
    fingeprints.append(data['bcutv5'])
    fingeprints.append(data['bcutv6'])
    fingeprints.append(data['bcutv7'])
    fingeprints.append(data['bcutv8'])
    fingeprints.append(data['bcutv9'])
    fingeprints.append(data['bcutv10'])
    fingeprints.append(data['bcutv11'])
    fingeprints.append(data['bcutv12'])
    fingeprints.append(data['bcutv13'])
    fingeprints.append(data['bcutv14'])
    fingeprints.append(data['bcutv15'])
    fingeprints.append(data['bcutv16'])

    fingeprints.append(data['LogP'])
    fingeprints.append(data['LogP2'])
    fingeprints.append(data['TPSA'])
    fingeprints.append(data['UI'])
    fingeprints.append(data['Hy'])
    fingeprints.append(data['IDET'])
    fingeprints.append(data['IDE'])
    fingeprints.append(data['IVDE'])
    fingeprints.append(data['Sitov'])
    fingeprints.append(data['Hatov'])
    fingeprints.append(data['Getov'])
    fingeprints.append(data['Gravto'])
    fingeprints.append(data['MR'])
    fingeprints.append(data['GMTIV'])

    fingeprints.append(data['PEOEVSA0'])
    fingeprints.append(data['PEOEVSA1'])
    fingeprints.append(data['PEOEVSA2'])
    fingeprints.append(data['PEOEVSA3'])
    fingeprints.append(data['PEOEVSA4'])
    fingeprints.append(data['PEOEVSA5'])
    fingeprints.append(data['PEOEVSA6'])
    fingeprints.append(data['PEOEVSA7'])
    fingeprints.append(data['PEOEVSA8'])
    fingeprints.append(data['PEOEVSA9'])
    fingeprints.append(data['PEOEVSA10'])
    fingeprints.append(data['PEOEVSA11'])

    fingeprints.append(data['MRVSA0'])
    fingeprints.append(data['MRVSA1'])
    fingeprints.append(data['MRVSA2'])
    fingeprints.append(data['MRVSA3'])
    fingeprints.append(data['MRVSA4'])
    fingeprints.append(data['MRVSA5'])
    fingeprints.append(data['MRVSA6'])
    fingeprints.append(data['MRVSA7'])
    fingeprints.append(data['MRVSA8'])
    fingeprints.append(data['MRVSA9'])

    fingeprints.append(data['VSAEstate0'])
    fingeprints.append(data['VSAEstate1'])
    fingeprints.append(data['VSAEstate2'])
    fingeprints.append(data['VSAEstate3'])
    fingeprints.append(data['VSAEstate4'])
    fingeprints.append(data['VSAEstate5'])
    fingeprints.append(data['VSAEstate6'])
    fingeprints.append(data['VSAEstate7'])
    fingeprints.append(data['VSAEstate8'])
    fingeprints.append(data['VSAEstate9'])

    fingeprints.append(data['slogPVSA0'])
    fingeprints.append(data['slogPVSA1'])
    fingeprints.append(data['slogPVSA2'])
    fingeprints.append(data['slogPVSA3'])
    fingeprints.append(data['slogPVSA4'])
    fingeprints.append(data['slogPVSA5'])
    fingeprints.append(data['slogPVSA6'])
    fingeprints.append(data['slogPVSA7'])
    fingeprints.append(data['slogPVSA8'])
    fingeprints.append(data['slogPVSA9'])
    fingeprints.append(data['slogPVSA10'])
    fingeprints.append(data['slogPVSA11'])

    return fingeprints