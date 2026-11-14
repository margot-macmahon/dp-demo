import os
import sys
import timeit
from functools import partial
from multiprocessing import Pool
import glob

import numpy as np
import phasespace
import tensorflow as tf
import trimesh
import uproot
from awkward.highlevel import Array as awk

# we focus on LAr ND but the code also produces results for GAr


def make_mesh(theta_oa):
    ND_baseline = 574 * 1e2  # distance to ND
    # 20.9*1e-3#52.26*1e-3#off-axis angle in rad
    dim_NDLAr = np.array(
        [350, 300, 500]
    )  # dimensions LAr ND however we make its width  narrower such that we only accept the neutrinos going through the center of the detector
    # dim_NDLAr = np.array([700, 300, 500]) #* u.cm # [x, y, z], cuboid shape, see DUNE ND CDR, sec. 2.1
    # dim_NDGAr = np.array([500, 260])   #   * u.cm # [x, r], horizontal cylindrical shape, see DUNE ND CDR, sec. 3.3.1
    TLAr = trimesh.transformations.translation_matrix(
        [ND_baseline * np.tan(theta_oa), 0, ND_baseline + 250]
    )  # translation to off-axis location
    NDLAr_mesh = trimesh.creation.box(dim_NDLAr, transform=TLAr)  # create a box
    # NDGAr_mesh = trimesh.creation.cylinder(radius=dim_NDGAr[1], height=dim_NDGAr[0], sections=50,transform=trimesh.transformations.concatenate_matrices(TLAr,trimesh.transformations.translation_matrix([0,0,8.61*1e2]), trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0])))
    return NDLAr_mesh  # , NDGAr_mesh


# meson masses in GeV
mPi = 139.57039 * 1e-3
mPi0 = 134.9768 * 1e-3
mkaon = 493.677 * 1e-3
mkaonL = 497.611 * 1e-3
mmu = 105.6583755 * 1e-3
me = 0.511 * 1e-3
mnu = 0


# define decays in phasespace
def twobodydecay(parentmass, daughtermass):
    fct = phasespace.nbody_decay(
        parentmass, [daughtermass, mnu], top_name="p", names=["CL", "nu"]
    )
    return fct


def threebodydecay(parentmass, daughtermass1, daughtermass2):
    fct = phasespace.nbody_decay(
        parentmass,
        [daughtermass1, daughtermass2, mnu],
        top_name="p",
        names=["p1", "p2", "nu"],
    )
    return fct


# calculate the 4-momenta  of the mesons which are needed for the boost in phasespace
def makeboostlist(decayid, parentmass, branches):
    lee = len(branches["decay.ndecay"])
    listp4 = []
    for i in range(0, lee):
        if branches["decay.ndecay"][i] == decayid:
            pxx = branches["decay.pdpx"][i]
            pyy = branches["decay.pdpy"][i]
            pzz = branches["decay.pdpz"][i]
            listp4.append(
                [pxx, pyy, pzz, np.sqrt(pxx**2 + pyy**2 + pzz**2 + parentmass**2)]
            )  # calculate the energy of the mesons by hand
    listp4arrtf = tf.convert_to_tensor(listp4)
    nev = len(listp4)
    return nev, listp4arrtf, listp4


# decay mesons with phasespace
def makeparticlelist(decayid, branches):
    if decayid == 13 or decayid == 14:
        CLmass = mmu  # pi to mu
        parentmass = mPi
        decay = twobodydecay(parentmass, CLmass)
    if decayid == 11 or decayid == 12:
        CLmass = me  # mu to e
        parentmass = mmu
        CLmass2 = 0
        decay = threebodydecay(parentmass, CLmass, CLmass2)
    if decayid == 8 or decayid == 5:
        CLmass = mmu  # k to mu
        parentmass = mkaon
        decay = twobodydecay(parentmass, CLmass)
    if decayid == 10 or decayid == 7:
        CLmass = mmu  # k to mu pi0
        CLmass2 = mPi0
        parentmass = mkaon
        decay = threebodydecay(parentmass, CLmass, CLmass2)
    if decayid == 9 or decayid == 6:
        CLmass = me  # k to e pi0
        CLmass2 = mPi0
        parentmass = mkaon
        decay = threebodydecay(parentmass, CLmass, CLmass2)
    if decayid == 1 or decayid == 2:
        CLmass1 = me  # Kl to e pi
        CLmass2 = mPi
        parentmass = mkaonL
        decay = threebodydecay(parentmass, CLmass1, CLmass2)
    if decayid == 3 or decayid == 4:
        CLmass1 = mmu  # kl to mu pi
        CLmass2 = mPi
        parentmass = mkaonL
        decay = threebodydecay(parentmass, CLmass1, CLmass2)

    nev, listboost, ll = makeboostlist(decayid, parentmass, branches)
    weights, particles = decay.generate(
        n_events=nev, boost_to=listboost, normalize_weights=True
    )
    pnu = particles["nu"]  # get neutrinos, not interested in produced charged leptons
    p1 = tf.make_tensor_proto(pnu)
    pf = tf.make_ndarray(p1)
    wnu = weights  # get weights of neutrinos, all 1
    w1 = tf.make_tensor_proto(wnu)
    wf = tf.make_ndarray(w1)
    return awk(pf), awk(ll), awk(wf)


# get vertex position in cm, MCzero is coordinate origin
def makevertexlist(decayid, branches):
    lee = len(branches["decay.ndecay"])
    listv3 = []
    for i in range(0, lee):
        if branches["decay.ndecay"][i] == decayid:
            vxx = branches["decay.vx"][i]
            vyy = branches["decay.vy"][i]
            vzz = branches["decay.vz"][i]
            listv3.append([vxx, vyy, vzz])
    return listv3


# get meson weights
def makeweightlist(decayid, branches):
    lee = len(branches["decay.ndecay"])
    listw = []
    for i in range(0, lee):
        if branches["decay.ndecay"][i] == decayid:
            wgt = branches["decay.nimpwt"][i]
            listw.append(wgt)
    return listw


# use momentum 4-vector to calculate trajectory of neutrinos
def makeraylist(momentumlist):
    lee = len(momentumlist)
    listnray = []
    for i in range(0, lee):
        absp = np.sqrt(
            momentumlist[i, 0] ** 2 + momentumlist[i, 1] ** 2 + momentumlist[i, 2] ** 2
        )
        listnray.append(
            [
                momentumlist[i, 0] / absp,
                momentumlist[i, 1] / absp,
                momentumlist[i, 2] / absp,
            ]
        )
    return listnray


# just checking if decays are valid
def decayvalid(decayid, branches):
    lee = len(branches["decay.ndecay"])
    i = 0
    for i in range(0, lee):
        if branches["decay.ndecay"][i] == decayid:
            i = i + 1
    return i


# remove all duplicates
def list_duplicates(seq):
    # make list of elements which go into the detector and out
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = {x for x in seq if x in seen or seen_add(x)}
    # turn the set into a list (as requested)
    return list(seen_twice)


def grandparents(decayid, branches):
    if decayid == 11 or decayid == 12:
        branch_d = branches[branches["decay.ndecay"] == decayid]
        return branch_d[
            "ancestor.pdg"
        ][
            :, -3
        ]  # last array element is neutrino, penultimate is neutrino parent, next back is grandparent, etc.


# main function which writes out energy and weights of neutrinos at off-axis locations
def filesout(
    theta_oa=0.0,
    NDLAr_mesh=None,
    fileid=0,
    is_rhc=False,
    alt_str="nom",
    in_dir="",
    out_dir="/unix/dune2/mmacmahon/nd/",
    decayid=0,
    n_rep=1,
):
    if decayid == 11 or decayid == 12:
        is_mu = True
        print("is_mu")
    else:
        is_mu = False

    # dir = "/unix/dune2/mmacmahon/qgsp/40_angles_80_repeats/rem/" # MUST HAVE / at the end of string!!!
    print("output directory = ", out_dir)
    prefix = f"{out_dir}/{str(fileid)}_thetaoa{str(theta_oa)}_decayid{str(decayid)}"
    if decayid == 11 or decayid == 12:
        prefix_pi = (
            f"{out_dir}/{str(fileid)}_thetaoa{str(theta_oa)}_decayid{str(decayid)}pi"
        )
        prefix_k = (
            f"{out_dir}/{str(fileid)}_thetaoa{str(theta_oa)}_decayid{str(decayid)}kp"
        )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    listNDLArEf = []
    listNDLArwf = []

    listNDLArposf = []

    if is_mu:
        listNDLArEf_pi = []
        listNDLArwf_pi = []
        listNDLArposf_pi = []
        listNDLArEf_k = []
        listNDLArwf_k = []
        listNDLArposf_k = []

    # there are up to 25 root files
    f_prefix = f"{in_dir}/g4lbne_v3r5p10-22-gf42572f_QGSP_BERT_OfficialEngDesignSept2021_neutrino_"

    if alt_str != "nom":
        if alt_str == "ftfp":
            f_prefix = f"{in_dir}/g4lbne_v3r5p10-23-g97c8e62_FTFP_BERT_OfficialEngDesignSept2021_neutrino_"
        elif alt_str == "qgsp":
            f_prefix = f"{in_dir}/g4lbne_v3r5p10-22-gf42572f_QGSP_BERT_OfficialEngDesignSept2021_neutrino_"
        else:
            print("Flux file type not recognised")

    if is_rhc:
        print("is_rhc")
        f_prefix = f"{in_dir}/rhc/g4lbne_v3r5p6_QGSP_BERT_OptEngNov2017_150cmTargetCone_NoMod2_antineutrino_"
        if alt_str != "nom":
            if alt_str == "ftfp":
                f_prefix = f"{in_dir}/ftfp-bert/rhc/g4lbne_v3r5p10-23-g97c8e62_FTFP_BERT_OfficialEngDesignSept2021_antineutrino_"
            elif alt_str == "qgsp":
                f_prefix = f"{in_dir}/qgsp_bert/rhc/g4lbne_v3r5p10-23-g97c8e62_QGSP_BERT_OfficialEngDesignSept2021_antineutrino_"

    root_files = sorted(glob.glob(f_prefix + "*.dk2nu.root"))
    if not root_files:
        raise FileNotFoundError(f"No dk2nu files for prefix {f_prefix}")
    for file_path in root_files:
        n_repeats = 1
        x = n_rep
        print(file_path)
        file = uproot.open(file_path)
        tree = file["dk2nuTree"]
        branches = tree.arrays()
        v = makevertexlist(decayid, branches)
        w = makeweightlist(decayid, branches)
        if is_mu:
            par = grandparents(decayid, branches)

        for j in range(x):
            n_repeats += 1
            start = timeit.default_timer()

            p, _ll, _wf = makeparticlelist(decayid, branches)

            n = makeraylist(p)

            X_array = v
            n_array = n

            NDLAr_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(NDLAr_mesh)
            NDLAr_entry_vertices, NDLAr_contained_HNLs2, _NDLAr_entry_faces = (
                NDLAr_intersector.intersects_location(
                    ray_origins=X_array, ray_directions=n_array
                )
            )

            NDLAr_contained_HNLs = list_duplicates(NDLAr_contained_HNLs2)

            listNDLArz = NDLAr_entry_vertices

            listNDLArE = []
            listNDLArw = []

            if is_mu:
                listNDLArE_pi = []
                listNDLArE_k = []
                listNDLArw_pi = []
                listNDLArw_k = []

                NDLAr_contained_HNLs_pi = awk(NDLAr_contained_HNLs)[
                    abs(par[NDLAr_contained_HNLs]) == 211
                ]
                NDLAr_contained_HNLs_k = awk(NDLAr_contained_HNLs)[
                    abs(par[NDLAr_contained_HNLs]) == 321
                ]
                listNDLArz_pi = awk(listNDLArz)
                listNDLArz_k = awk(listNDLArz)

                for i in NDLAr_contained_HNLs_pi:
                    listNDLArE_pi.append(p[i, 3])
                    listNDLArw_pi.append(w[i])
                for i in NDLAr_contained_HNLs_k:
                    listNDLArE_k.append(p[i, 3])
                    listNDLArw_k.append(w[i])

                listNDLArEf_pi.extend(listNDLArE_pi)
                listNDLArwf_pi.extend(listNDLArw_pi)
                listNDLArEf_k.extend(listNDLArE_k)
                listNDLArwf_k.extend(listNDLArw_k)
                listNDLArposf_pi.extend(listNDLArz_pi)
                listNDLArposf_k.extend(listNDLArz_k)

            else:
                for i in NDLAr_contained_HNLs:
                    listNDLArE.append(p[i, 3])
                    listNDLArw.append(w[i])

                listNDLArEf.extend(listNDLArE)
                listNDLArwf.extend(listNDLArw)
                listNDLArposf.extend(listNDLArz)
            print(file_path, "root # ")
            print("time = ", timeit.default_timer() - start)

    print("n_repeats = ", n_repeats)

    if is_mu:
        contentEL_pi = str(listNDLArEf_pi)
        contentwL_pi = str(listNDLArwf_pi)
        contentposL_pi = str(listNDLArposf_pi)

        contentEL_k = str(listNDLArEf_k)
        contentwL_k = str(listNDLArwf_k)
        contentposL_k = str(listNDLArposf_k)

        fileoutEL_pi = open(
            prefix_pi + "_NDLAr_energy_smallND2_" + alt_str + ".txt", "a"
        )
        fileoutwL_pi = open(
            prefix_pi + "_NDLAr_weights_smallND2_" + alt_str + ".txt", "a"
        )
        fileoutposL_pi = open(
            prefix_pi + "_NDLAr_entries_smallND2_" + alt_str + ".txt", "a"
        )

        fileoutEL_k = open(prefix_k + "_NDLAr_energy_smallND2_" + alt_str + ".txt", "a")
        fileoutwL_k = open(
            prefix_k + "_NDLAr_weights_smallND2_" + alt_str + ".txt", "a"
        )
        fileoutposL_k = open(
            prefix_k + "_NDLAr_entries_smallND2_" + alt_str + ".txt", "a"
        )

        fileoutEL_pi.write(contentEL_pi)
        fileoutwL_pi.write(contentwL_pi)
        fileoutposL_pi.write(contentposL_pi)

        fileoutEL_k.write(contentEL_k)
        fileoutwL_k.write(contentwL_k)
        fileoutposL_k.write(contentposL_k)

        fileoutEL_pi.close()
        fileoutwL_pi.close()
        fileoutposL_pi.close()

        fileoutEL_k.close()
        fileoutwL_k.close()
        fileoutposL_k.close()

    else:
        contentEL = str(listNDLArEf)
        contentwL = str(listNDLArwf)

        contentposL = str(listNDLArposf)

        fileoutEL = open(prefix + "_NDLAr_energy_smallND2_" + alt_str + ".txt", "a")
        fileoutwL = open(prefix + "_NDLAr_weights_smallND2_" + alt_str + ".txt", "a")

        fileoutposL = open(prefix + "_NDLAr_entries_smallND_" + alt_str + ".txt", "a")

        fileoutEL.write(contentEL)
        fileoutwL.write(contentwL)

        fileoutposL.write(contentposL)

        fileoutEL.close()
        fileoutwL.close()
        fileoutposL.close()

    return 0


def process_decay_id(params):
    decay_id, kwargs = params
    return filesout(decayid=decay_id, **kwargs)


if __name__ == "__main__":
    # TODO use argparse instead of raw sys.argv
    thetai = float(sys.argv[1])
    fileid = int(sys.argv[2])
    id_all = eval(sys.argv[3])
    is_rhc = eval(sys.argv[5])
    alt_str = str(sys.argv[6])
    in_dir = str(sys.argv[7])
    out_dir = str(sys.argv[8])

    if is_rhc:
        print("is_rhc")
    if alt_str != "nom":
        print(alt_str)

    theta = np.array([thetai])  # ,0.0209,0.03136,0.04181,0.05226])
    print(theta)
    decayid = np.arange(1, 15)

    meshes = make_mesh(thetai)
    if id_all:
        kwargs = dict(
            theta_oa=thetai,
            NDLAr_mesh=meshes,
            fileid=fileid,
            is_rhc=is_rhc,
            alt_str=alt_str,
            in_dir=in_dir,
            out_dir=out_dir,
        )
        print("theta = ", thetai, "decay = ", decayid)
        params = [(int(d), kwargs) for d in decayid]

        with Pool(min(len(decayid), os.cpu_count())) as p:
            p.map(process_decay_id, params)
    else:
        dec_id = int(sys.argv[4])
        filesout(
            theta_oa=thetai,
            NDLAr_mesh=meshes,
            fileid=fileid,
            is_rhc=is_rhc,
            alt_str=alt_str,
            decayid=dec_id,
            in_dir=in_dir,
            out_dir=out_dir,
        )
        print("running theta = ", theta, "decay id = ", dec_id)
