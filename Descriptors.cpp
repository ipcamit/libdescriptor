#include "Descriptors.hpp"

int enzyme_dup, enzyme_out, enzyme_const;

template<typename T>
T __enzyme_virtualreverse(T);

void __enzyme_integer(int*,size_t);

// Rev mode diff
void __enzyme_autodiff(void (*)(int, int *, int *, int *, double *, double *, DescriptorKind *),
                       int, int /* n_atoms */,
                       int, int * /* Z */,
                       int, int * /* neighbor list */,
                       int, int * /* number_of_neigh_list */,
                       int, double * /* coordinates */, double * /* derivative w.r.t coordinates */,
                       int, double * /* zeta */, double * /* dzeta_dE */,
                       int, DescriptorKind * /* DescriptorKind to diff */, DescriptorKind * /* d_DescriptorKind */);

void __enzyme_autodiff_batch(void (*)(int, int *, int *, int *, int *, int *, double *, double *, DescriptorKind *),
                       int, int, /* n_configs */
                       int, int *, /* n_atoms */
                       int, int *, /* ptr */
                       int, int *, /* Z */
                       int, int *, /* neighbor list */
                       int, int *, /* number_of_neigh_list */
                       int, double * /* coordinates */, double * /* derivative w.r.t coordinates */,
                       int, double * /* zeta */, double * /* dzeta_dE */,
                       int, DescriptorKind * /* DescriptorKind to diff */, DescriptorKind * /* d_DescriptorKind */);

void __enzyme_autodiff_one_atom(void (*)(int, int, int *, int *, int, double *, double *, DescriptorKind *),
                                int, int,
                                int, int /* n_atoms */,
                                int, int * /* Z */,
                                int, int * /* neighbor list */,
                                int, int  /* number_of_neigh_list */,
                                int, double * /* coordinates */, double * /* derivative w.r.t coordinates */,
                                int, double * /* zeta */, double * /* dzeta_dE */,
                                int, DescriptorKind * /* DescriptorKind to diff */,
                                DescriptorKind * /* d_DescriptorKind */);

/* Fwd mode AD. As Descriptors are many-to-many functions, in certain situations,
 * like in case of global descriptors, FWD mode might be more performant.
 * Also in future when we will have full jacobian of descriptors enabled,
 * FWD mode might be used. So adding it now only as a simple to-be-used in future.
 */
//TODO
void __enzyme_fwddiff(void (*)(int, int *, int *, int *, double *, double *, DescriptorKind *),
                      int, int /* n_atoms */,
                      int, int * /* Z */,
                      int, int * /* neighbor list */,
                      int, int * /* number_of_neigh_list */,
                      int, double * /* coordinates */, double * /* derivative w.r.t coordinates */,
                      int, double * /* zeta */, double * /* dzeta_dE */,
                      int, DescriptorKind * /* DescriptorKind to diff */, DescriptorKind * /* d_DescriptorKind */);

void __enzyme_fwddiff_one_atom(void (*)(int, int, int *, int *, int, double *, double *, DescriptorKind *),
                               int, int,
                               int, int /* n_atoms */,
                               int, int * /* Z */,
                               int, int * /* neighbor list */,
                               int, int  /* number_of_neigh_list */,
                               int, double * /* coordinates */, double * /* derivative w.r.t coordinates */,
                               int, double * /* zeta */, double * /* dzeta_dE */,
                               int, DescriptorKind * /* DescriptorKind to diff */,
                               DescriptorKind * /* d_DescriptorKind */);

using namespace Descriptor;

DescriptorKind *DescriptorKind::initDescriptor(std::string &descriptor_file_name,
                                               AvailableDescriptor descriptor_kind) {
    if (descriptor_kind == KindSymmetryFunctions) {
        auto sf = new SymmetryFunctions(descriptor_file_name);
        sf->descriptor_kind = descriptor_kind;
        sf->descriptor_param_file = descriptor_file_name;
        return sf;
    } else if (descriptor_kind == KindBispectrum) {
        auto bs = new Bispectrum(descriptor_file_name);
        bs->descriptor_kind = descriptor_kind;
        bs->descriptor_param_file = descriptor_file_name;
        return bs;
    } else if (descriptor_kind == KindSOAP) {
        auto global = new SOAP(descriptor_file_name);
        global->descriptor_kind = descriptor_kind;
        global->descriptor_param_file = descriptor_file_name;
        return global;
//    } else if (descriptor_kind == KindXi) {
//        auto xi = new Xi(descriptor_file_name);
//        xi->descriptor_kind = descriptor_kind;
//        xi->descriptor_param_file = descriptor_file_name;
//        return xi;
    } else {
        throw std::invalid_argument("Descriptor kind not implemented yet");
    }
}

DescriptorKind *DescriptorKind::initDescriptor(AvailableDescriptor descriptor_kind) {
    if (descriptor_kind == KindSymmetryFunctions) {
        return new SymmetryFunctions();
    } else if (descriptor_kind == KindBispectrum) {
        return new Bispectrum();
    } else if (descriptor_kind == KindSOAP) {
        return new SOAP();
//    } else if (descriptor_kind == KindXi) {
//        return new Xi();
    } else {
        throw std::invalid_argument("Descriptor kind not implemented yet");
    }
}

/* This is wrapper for generic descriptor class compute function
   This wrapper will be differentiated using enzyme for more generic
   library structure. */
void Descriptor::compute(int const n_atoms /* contributing */,
                         int *const species,
                         int *const neighbor_list,
                         int *const number_of_neighbors,
                         double *const coordinates,
                         double *const desc,
                         DescriptorKind *const desc_kind) {
    int *neighbor_ptr = neighbor_list;
    double *desc_ptr = desc;
    for (int i = 0; i < n_atoms; i++) {
        desc_kind->compute(i, n_atoms, species, neighbor_ptr, number_of_neighbors[i],
                           coordinates, desc_ptr);
        neighbor_ptr += number_of_neighbors[i];
        desc_ptr += desc_kind->width;
    }
}

void Descriptor::compute_batch(int n_configurations, int *n_atoms, int *ptr, int *species, int *neighbor_list,
                               int *number_of_neighs, double *coordinates, double *desc,
                               Descriptor::DescriptorKind *descriptor_kind) {
    int *neighbor_ptr = neighbor_list;
    double *desc_ptr = desc;
    int idx = 0;
    int total_atoms = 0;
    for (int i = 0; i < n_configurations; i++) {
        total_atoms += n_atoms[i];
    }

    int total_neighs = 0;
    for (int i = 0; i < total_atoms; i++) {
        total_neighs += number_of_neighs[i];
    }

    for (int i = 0; i < n_configurations; i++) {
        for (int j = 0; j < n_atoms[i]; j++) {
            descriptor_kind->compute(j + ptr[i], total_atoms, species, neighbor_ptr, number_of_neighs[idx],
                                     coordinates, desc_ptr);
            neighbor_ptr += number_of_neighs[idx];
            desc_ptr += descriptor_kind->width;
            idx++;
        }
    }
}

void Descriptor::gradient(int n_atoms /* contributing */,
                          int *species,
                          int *neighbor_list,
                          int *number_of_neighbors,
                          double *coordinates,
                          double *d_coordinates,
                          double *desc,
                          double *d_desc, /* vector for vjp or jvp */
                          DescriptorKind *desc_kind) {
    switch (desc_kind->descriptor_kind) {
        case KindSymmetryFunctions: {
            auto d_desc_kind = new SymmetryFunctions();
            *((void **) d_desc_kind) = __enzyme_virtualreverse(*((void **) d_desc_kind));

            d_desc_kind->clone_empty(desc_kind);
            __enzyme_autodiff(compute, /* fn to be differentiated */
                              enzyme_const, n_atoms, /* Do not diff. against integer params */
                              enzyme_const, species,
                              enzyme_const, neighbor_list,
                              enzyme_const, number_of_neighbors,
                              enzyme_dup, coordinates, d_coordinates,
                              enzyme_dup, desc, d_desc,
                              enzyme_dup, desc_kind, d_desc_kind);
            delete d_desc_kind;
            return;
        }
        case KindBispectrum: {
            auto d_desc_kind = new Bispectrum();
            *((void **) d_desc_kind) = __enzyme_virtualreverse(*((void **) d_desc_kind));

            d_desc_kind->clone_empty(desc_kind);
            __enzyme_autodiff(compute, /* fn to be differentiated */
                              enzyme_const, n_atoms, /* Do not diff. against integer params */
                              enzyme_const, species,
                              enzyme_const, neighbor_list,
                              enzyme_const, number_of_neighbors,
                              enzyme_dup, coordinates, d_coordinates,
                              enzyme_dup, desc, d_desc,
                              enzyme_dup, desc_kind, d_desc_kind);
            delete d_desc_kind;
            return;
        }
        case KindSOAP: {
            auto d_desc_kind = new SOAP();
            *((void **) d_desc_kind) = __enzyme_virtualreverse(*((void **) d_desc_kind));

            d_desc_kind->clone_empty(desc_kind);
            __enzyme_autodiff(compute, /* fn to be differentiated */
                              enzyme_const, n_atoms, /* Do not diff. against integer params */
                              enzyme_const, species,
                              enzyme_const, neighbor_list,
                              enzyme_const, number_of_neighbors,
                              enzyme_dup, coordinates, d_coordinates,
                              enzyme_dup, desc, d_desc,
                              enzyme_dup, desc_kind, d_desc_kind);
            // int *neighbor_ptr = neighbor_list;
            // double *desc_ptr = desc;
            // double *d_desc_ptr = d_desc;
            // for (int i = 0; i < n_atoms; i++) {
            //     __enzyme_autodiff_one_atom(compute_single_atom, /* fn to be differentiated */
            //                enzyme_const, i,
            //                enzyme_const, n_atoms, /* Do not diff. against integer params */
            //                enzyme_const, species,
            //                enzyme_const, neighbor_ptr,
            //                enzyme_const, number_of_neighbors[i],
            //                enzyme_dup, coordinates, d_coordinates,
            //                enzyme_dup, desc_ptr, d_desc_ptr,
            //                enzyme_dup, desc_kind, d_desc_kind);
            //     neighbor_ptr += number_of_neighbors[i];
            //     desc_ptr += desc_kind->width;
            //     d_desc_ptr += d_desc_kind->width;
            // }
            delete d_desc_kind;
            return;
        }
//        case KindXi: {
//            auto d_desc_kind = new Xi();
//            *((void **) d_desc_kind) = __enzyme_virtualreverse(*((void **) d_desc_kind));
//
//            d_desc_kind->clone_empty(desc_kind);
//            __enzyme_autodiff(compute, /* fn to be differentiated */
//                              enzyme_const, n_atoms, /* Do not diff. against integer params */
//                              enzyme_const, species,
//                              enzyme_const, neighbor_list,
//                              enzyme_const, number_of_neighbors,
//                              enzyme_dup, coordinates, d_coordinates,
//                              enzyme_dup, desc, d_desc,
//                              enzyme_dup, desc_kind, d_desc_kind);
//            delete d_desc_kind;
//            return;
//        }
        default:
            std::cerr << "Descriptor kind not supported\n";
            throw std::invalid_argument("Descriptor kind not supported");
    }
}


void Descriptor::gradient_batch(int n_configurations,
                                int *n_atoms,
                                int *ptr,
                                int *species,
                                int *neighbor_list,
                                int *number_of_neighs,
                                double *coordinates,
                                double *d_coordinates,
                                double *desc,
                                double *d_desc,
                                Descriptor::DescriptorKind *descriptor_to_diff) {
    switch (descriptor_to_diff->descriptor_kind) {
        case KindSymmetryFunctions: {
            auto d_desc_kind = new SymmetryFunctions();
            *((void **) d_desc_kind) = __enzyme_virtualreverse(*((void **) d_desc_kind));

            d_desc_kind->clone_empty(descriptor_to_diff);
            __enzyme_autodiff_batch(compute_batch, /* fn to be differentiated */
                              enzyme_const, n_configurations, /* Do not diff. against integer params */
                              enzyme_const, n_atoms,
                              enzyme_const, ptr,
                              enzyme_const, species,
                              enzyme_const, neighbor_list,
                              enzyme_const, number_of_neighs,
                              enzyme_dup, coordinates, d_coordinates,
                              enzyme_dup, desc, d_desc,
                              enzyme_dup, descriptor_to_diff, d_desc_kind);
            delete d_desc_kind;
            return;
        }
        case KindBispectrum: {
            auto d_desc_kind = new Bispectrum();
            *((void **) d_desc_kind) = __enzyme_virtualreverse(*((void **) d_desc_kind));

            d_desc_kind->clone_empty(descriptor_to_diff);
            __enzyme_autodiff_batch(compute_batch, /* fn to be differentiated */
                              enzyme_const, n_configurations, /* Do not diff. against integer params */
                              enzyme_const, n_atoms,
                              enzyme_const, ptr,
                              enzyme_const, species,
                              enzyme_const, neighbor_list,
                              enzyme_const, number_of_neighs,
                              enzyme_dup, coordinates, d_coordinates,
                              enzyme_dup, desc, d_desc,
                              enzyme_dup, descriptor_to_diff, d_desc_kind);
            delete d_desc_kind;
            return;
            return;
        }
        case KindSOAP: {
            auto d_desc_kind = new SOAP();
            *((void **) d_desc_kind) = __enzyme_virtualreverse(*((void **) d_desc_kind));

            d_desc_kind->clone_empty(descriptor_to_diff);
            __enzyme_autodiff_batch(compute_batch, /* fn to be differentiated */
                              enzyme_const, n_configurations, /* Do not diff. against integer params */
                              enzyme_const, n_atoms,
                              enzyme_const, ptr,
                              enzyme_const, species,
                              enzyme_const, neighbor_list,
                              enzyme_const, number_of_neighs,
                              enzyme_dup, coordinates, d_coordinates,
                              enzyme_dup, desc, d_desc,
                              enzyme_dup, descriptor_to_diff, d_desc_kind);
            delete d_desc_kind;
            return;
        }
//        case KindXi: {
//            auto d_desc_kind = new Xi();
//            *((void **) d_desc_kind) = __enzyme_virtualreverse(*((void **) d_desc_kind));
//
//            d_desc_kind->clone_empty(descriptor_to_diff);
//            __enzyme_autodiff_batch(compute_batch, /* fn to be differentiated */
//                              enzyme_const, n_configurations, /* Do not diff. against integer params */
//                              enzyme_const, n_atoms,
//                              enzyme_const, ptr,
//                              enzyme_const, species,
//                              enzyme_const, neighbor_list,
//                              enzyme_const, number_of_neighs,
//                              enzyme_dup, coordinates, d_coordinates,
//                              enzyme_dup, desc, d_desc,
//                              enzyme_dup, descriptor_to_diff, d_desc_kind);
//            delete d_desc_kind;
//            return;
//        }
        default:
            std::cerr << "Descriptor kind not supported\n";
            throw std::invalid_argument("Descriptor kind not supported");
    }
}


void Descriptor::compute_single_atom(int index,
                                     int const n_atoms /* contributing */,
                                     int *const species,
                                     int *const neighbor_list,
                                     int number_of_neighbors,
                                     double *const coordinates,
                                     double *const desc,
                                     DescriptorKind *const desc_kind) {
    desc_kind->compute(index, n_atoms, species, neighbor_list, number_of_neighbors,
                       coordinates, desc);
}


void Descriptor::gradient_single_atom(int index,
                                      int n_atoms /* contributing */,
                                      int *species,
                                      int *neighbor_list,
                                      int number_of_neighbors,
                                      double *coordinates,
                                      double *d_coordinates,
                                      double *desc,
                                      double *d_desc, /* vector for vjp or jvp */
                                      DescriptorKind *desc_kind) {
    switch (desc_kind->descriptor_kind) {
        case KindSymmetryFunctions: {
            auto d_desc_kind = new SymmetryFunctions();
            *((void **) d_desc_kind) = __enzyme_virtualreverse(*((void **) d_desc_kind));
            d_desc_kind->clone_empty(desc_kind);

            __enzyme_autodiff_one_atom(compute_single_atom, /* fn to be differentiated */
                                       enzyme_const, index,
                                       enzyme_const, n_atoms, /* Do not diff. against integer params */
                                       enzyme_const, species,
                                       enzyme_const, neighbor_list,
                                       enzyme_const, number_of_neighbors,
                                       enzyme_dup, coordinates, d_coordinates,
                                       enzyme_dup, desc, d_desc,
                                       enzyme_dup, desc_kind, d_desc_kind);
            delete d_desc_kind;
            return;
        }
        case KindBispectrum: {
            auto d_desc_kind = new Bispectrum();

            *((void **) d_desc_kind) = __enzyme_virtualreverse(*((void **) d_desc_kind));
            d_desc_kind->clone_empty(desc_kind);

            __enzyme_autodiff_one_atom(compute_single_atom, /* fn to be differentiated */
                                       enzyme_const, index,
                                       enzyme_const, n_atoms, /* Do not diff. against integer params */
                                       enzyme_const, species,
                                       enzyme_const, neighbor_list,
                                       enzyme_const, number_of_neighbors,
                                       enzyme_dup, coordinates, d_coordinates,
                                       enzyme_dup, desc, d_desc,
                                       enzyme_dup, desc_kind, d_desc_kind);
            delete d_desc_kind;
            return;
        }
        case KindSOAP: {
            auto d_desc_kind = new SOAP();

            *((void **) d_desc_kind) = __enzyme_virtualreverse(*((void **) d_desc_kind));
            d_desc_kind->clone_empty(desc_kind);
            __enzyme_autodiff_one_atom(compute_single_atom, /* fn to be differentiated */
                                       enzyme_const, index,
                                       enzyme_const, n_atoms, /* Do not diff. against integer params */
                                       enzyme_const, species,
                                       enzyme_const, neighbor_list,
                                       enzyme_const, number_of_neighbors,
                                       enzyme_dup, coordinates, d_coordinates,
                                       enzyme_dup, desc, d_desc,
                                       enzyme_dup, desc_kind, d_desc_kind);
            delete d_desc_kind;
            return;
        }
//        case KindXi: {
//            auto d_desc_kind = new Xi();
//
//            *((void **) d_desc_kind) = __enzyme_virtualreverse(*((void **) d_desc_kind));
//            d_desc_kind->clone_empty(desc_kind);
//            __enzyme_autodiff_one_atom(compute_single_atom, /* fn to be differentiated */
//                                       enzyme_const, index,
//                                       enzyme_const, n_atoms, /* Do not diff. against integer params */
//                                       enzyme_const, species,
//                                       enzyme_const, neighbor_list,
//                                       enzyme_const, number_of_neighbors,
//                                       enzyme_dup, coordinates, d_coordinates,
//                                       enzyme_dup, desc, d_desc,
//                                       enzyme_dup, desc_kind, d_desc_kind);
//            delete d_desc_kind;
//            return;
//        }
        default:
            std::cerr << "Descriptor kind not supported\n";
            throw std::invalid_argument("Descriptor kind not supported");
    }
}

void Descriptor::jacobian(int n_atoms, /* contributing */
                          int n_total_atoms,
                          int *species,
                          int *neighbor_list,
                          int *number_of_neighs,
                          double *coordinates,
                          double *J_coordinates,
                          DescriptorKind *descriptor_to_diff) {
    auto desc = new double[descriptor_to_diff->width * n_atoms];
    std::vector<double> d_desc;
    d_desc = std::vector<double>(descriptor_to_diff->width * n_atoms);
    std::fill(desc, desc + descriptor_to_diff->width * n_atoms, 0.0);
    // TODO: #pragma omp parallel for private(d_desc, desc)
    for (int j = 0; j < descriptor_to_diff->width * n_atoms; j++) {
        if ( j == 0 ) {
            d_desc[j] = 1.0;
        } else {
            d_desc[j - 1] = 0.0;
            d_desc[j] = 1.0;
        }
        gradient(n_atoms,
                 species,
                 neighbor_list,
                 number_of_neighs,
                 coordinates,
                 J_coordinates + j * 3 * n_total_atoms,
                 desc,
                 d_desc.data(),
                 descriptor_to_diff);
    }

    delete[] desc;
}

DescriptorKind::~DescriptorKind() = default;

DescriptorKind *
DescriptorKind::initDescriptor(AvailableDescriptor availableDescriptorKind, std::vector<std::string> *species,
                               std::string *cutoff_function, double *cutoff,
                               std::vector<std::string> *symmetry_function_types,
                               std::vector<int> *symmetry_function_sizes,
                               std::vector<double> *symmetry_function_parameters) {
    std::vector<double> cutoff_array(species->size() * species->size(), cutoff[0]);
    auto return_pointer = new SymmetryFunctions(species, cutoff_function, cutoff_array.data(),
                                                symmetry_function_types, symmetry_function_sizes,
                                                symmetry_function_parameters);
    return return_pointer;
}

DescriptorKind *
DescriptorKind::initDescriptor(AvailableDescriptor availableDescriptorKind, double rfac0_in, int twojmax_in,
                               int diagonalstyle_in,
                               int use_shared_arrays_in, double rmin0_in, int switch_flag_in, int bzero_flag_in,
                               double *cutoff, std::vector<std::string> *species, std::vector<double> *weights) {
    std::vector<double> cutoff_array(species->size() * species->size(), cutoff[0]);
    auto return_pointer = new Bispectrum(rfac0_in, twojmax_in, diagonalstyle_in,
                                         use_shared_arrays_in, rmin0_in, switch_flag_in, bzero_flag_in);
    return_pointer->width = return_pointer->get_width();
    return_pointer->set_species(species->size());
    std::string cutoff_function = "cos";
    return_pointer->set_cutoff(cutoff_function.c_str(), species->size(), cutoff_array.data());
    return_pointer->set_weight(species->size(), weights->data());
    return_pointer ->descriptor_kind = availableDescriptorKind;
    return return_pointer;
}

DescriptorKind *
DescriptorKind::initDescriptor(AvailableDescriptor availableDescriptorKind, int n_max, int l_max, double cutoff,
                               std::vector<std::string> &species, std::string radial_basis, double eta) {
    auto return_pointer = new SOAP(n_max, l_max, cutoff, species, radial_basis, eta);
    return_pointer->width = return_pointer->get_width();
    return_pointer->descriptor_kind = availableDescriptorKind;

    return return_pointer;
}

//DescriptorKind *
//DescriptorKind::initDescriptor(AvailableDescriptor availableDescriptorKind, int q, double cutoff,
//                               std::vector<std::string> &species, std::string& radial_basis) {
//    auto return_pointer = new Xi(q, cutoff, species, radial_basis);
//    return_pointer->width = return_pointer->get_width();
//    return_pointer->descriptor_kind = availableDescriptorKind;
//    return return_pointer;
//}

// Bispectrum.cpp
#define MY_PI 3.1415926535897932
#define DIM 3

Bispectrum::Bispectrum(double const rfac0_in,
                       int const twojmax_in,
                       int const diagonalstyle_in,
                       int const use_shared_arrays_in,
                       double const rmin0_in,
                       int const switch_flag_in,
                       int const bzero_flag_in) :
        nmax(0),// TODO Fixed nmax, modify it after enzyme bugfix
        twojmax(twojmax_in),
        diagonalstyle(diagonalstyle_in),
        rmin0(rmin0_in),
        rfac0(rfac0_in),
        use_shared_arrays(use_shared_arrays_in),
        switch_flag(switch_flag_in),
        wself(1.0),
        bzero_flag(bzero_flag_in) {
    ncoeff = compute_ncoeff();

    create_twojmax_arrays();

    if (bzero_flag) {
        double const www = wself * wself * wself;
        for (int j = 1; j <= twojmax + 1; ++j) { bzero[j] = www * j; }
    }

    // 1D
    bvec.resize(ncoeff, static_cast<double>(0));

    // 2D
    dbvec.resize(ncoeff, 3);

    build_indexlist();

    init();

    grow_rij(250); // TODO Fixed nmax, modify it after enzyme bugfix

    width = get_width();
}

void Bispectrum::build_indexlist() {
    switch (diagonalstyle) {
        case 0: {
            int idxj_count = 0;
            for (int j1 = 0; j1 <= twojmax; ++j1) {
                for (int j2 = 0; j2 <= j1; ++j2) {
                    for (int j = std::abs(j1 - j2); j <= std::min(twojmax, j1 + j2);
                         j += 2) {
                        idxj_count++;
                    }
                }
            }

            // indexList can be changed here
            idxj.resize(idxj_count);

            idxj_max = idxj_count;

            idxj_count = 0;
            for (int j1 = 0; j1 <= twojmax; ++j1) {
                for (int j2 = 0; j2 <= j1; ++j2) {
                    for (int j = std::abs(j1 - j2); j <= std::min(twojmax, j1 + j2);
                         j += 2) {
                        idxj[idxj_count].j1 = j1;
                        idxj[idxj_count].j2 = j2;
                        idxj[idxj_count].j = j;
                        idxj_count++;
                    }
                }
            }
            return;
        }
        case 1: {
            int idxj_count = 0;
            for (int j1 = 0; j1 <= twojmax; ++j1) {
                for (int j = 0; j <= std::min(twojmax, 2 * j1); j += 2) {
                    idxj_count++;
                }
            }

            // indexList can be changed here
            idxj.resize(idxj_count);

            idxj_max = idxj_count;

            idxj_count = 0;
            for (int j1 = 0; j1 <= twojmax; ++j1) {
                for (int j = 0; j <= std::min(twojmax, 2 * j1); j += 2) {
                    idxj[idxj_count].j1 = j1;
                    idxj[idxj_count].j2 = j1;
                    idxj[idxj_count].j = j;
                    idxj_count++;
                }
            }
            return;
        }
        case 2: {
            int idxj_count = 0;
            for (int j1 = 0; j1 <= twojmax; ++j1) { idxj_count++; }

            // indexList can be changed here
            idxj.resize(idxj_count);
            idxj_max = idxj_count;
            idxj_count = 0;

            for (int j1 = 0; j1 <= twojmax; ++j1) {
                idxj[idxj_count].j1 = j1;
                idxj[idxj_count].j2 = j1;
                idxj[idxj_count].j = j1;
                idxj_count++;
            }
            return;
        }
        case 3: {
            int idxj_count = 0;
            for (int j1 = 0; j1 <= twojmax; ++j1) {
                for (int j2 = 0; j2 <= j1; ++j2) {
                    for (int j = std::abs(j1 - j2); j <= std::min(twojmax, j1 + j2);
                         j += 2) {
                        if (j >= j1) { idxj_count++; }
                    }
                }
            }

            // indexList can be changed here
            idxj.resize(idxj_count);
            idxj_max = idxj_count;
            idxj_count = 0;

            for (int j1 = 0; j1 <= twojmax; ++j1) {
                for (int j2 = 0; j2 <= j1; ++j2) {
                    for (int j = std::abs(j1 - j2); j <= std::min(twojmax, j1 + j2);
                         j += 2) {
                        if (j >= j1) {
                            idxj[idxj_count].j1 = j1;
                            idxj[idxj_count].j2 = j2;
                            idxj[idxj_count].j = j;
                            idxj_count++;
                        }
                    }
                }
            }
            return;
        }
        default:
            std::cerr << "The input style index = " + std::to_string(diagonalstyle)
                         + " is not a valid index!";
            std::abort();
    }
}

void Bispectrum::init() {
    init_clebsch_gordan();

    init_rootpqarray();
}

double Bispectrum::memory_usage() {
    double bytes;
    int const jdim = twojmax + 1;
    bytes = jdim * jdim * jdim * jdim * jdim * sizeof(double);
    bytes += 2 * jdim * jdim * jdim * sizeof(std::complex<double>);
    bytes += 2 * jdim * jdim * jdim * sizeof(double);
    bytes += jdim * jdim * jdim * 3 * sizeof(std::complex<double>);
    bytes += jdim * jdim * jdim * 3 * sizeof(double);
    bytes += ncoeff * sizeof(double);
    bytes += jdim * jdim * jdim * jdim * jdim * sizeof(std::complex<double>);
    return bytes;
}

void Bispectrum::grow_rij(int const newnmax) {
    if (newnmax > nmax) {
        nmax = newnmax;
        if (!use_shared_arrays) {
            // 2D
            rij.resize(nmax, 3);

            // 1D
            inside.resize(nmax, 0);
            wj.resize(nmax, static_cast<double>(0));
            rcutij.resize(nmax, static_cast<double>(0));
        }
    }
}

void Bispectrum::compute(int const index,
                         int const n_atoms,
                         int *const species,
                         int *const neigh_list,
                         int const number_of_neigh,
                         double *const coords,
                         double *const desc) {
    // prepare data

    auto coordinates = (VectorOfSize3 *) coords;

    int const *const ilist = neigh_list;
    int const iSpecies = species[index];

    // grow_rij removed for enzyme
    // insure rij, inside, wj, and rcutij are of size jnum

    // rij[, 3] = displacements between atom I and those neighbors
    // inside = indices of neighbors of I within cutoff
    // wj = weights for neighbors of I within cutoff
    // rcutij = cutoffs for neighbors of I within cutoff
    // note Rij sign convention => dU/dRij = dU/dRj = -dU/dRi

    int ninside = 0;

    // Setup loop over neighbors of current particle
    for (int jj = 0; jj < number_of_neigh; ++jj) {
        // adjust index of particle neighbor
        int const j = ilist[jj];
        int const jSpecies = species[j];

        // rij vec and
        double rvec[DIM];
        for (int dim = 0; dim < DIM; ++dim) {
            rvec[dim] = coordinates[j][dim] - coordinates[index][dim];
        }

        double const rsq
                = rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2];
        double const rmag = std::sqrt(rsq);

        // if (rmag < rcuts[iSpecies * n_species + jSpecies] && rmag > 1e-10) {
        if (rmag < rcuts(iSpecies, jSpecies) && rmag > 1e-10) {
            // rij[ninside * 3 + 0] = rvec[0];
            // rij[ninside * 3 + 1] = rvec[1];
            // rij[ninside * 3 + 2] = rvec[2];
            rij(ninside, 0) = rvec[0];
            rij(ninside, 1) = rvec[1];
            rij(ninside, 2) = rvec[2];

            inside[ninside] = j;
            wj[ninside] = wjelem[jSpecies];
            // rcutij[ninside] = rcuts[iSpecies * n_species  + jSpecies];
            rcutij[ninside] = rcuts(iSpecies, jSpecies);

            ninside++;
        }
    }
    // compute Ui, Zi, and Bi for atom I
    compute_ui(ninside);
    compute_zi();
    compute_bi();
    copy_bi2bvec();

    for (int icoeff = 0; icoeff < ncoeff; icoeff++) {
        // Modified as the descriptor location shall not be a concern of compute function
        desc[icoeff] = bvec[icoeff];
    }
}

void Bispectrum::set_cutoff(const char *name,
                            std::size_t const Nspecies,
                            double const *rcuts_in) {
    // store number of species and cutoff values

    for (int i = 0; i < Nspecies * Nspecies; i++) {
        rcuts.push_back(*(rcuts_in + i));
    }
}

void Bispectrum::set_weight(int const Nspecies, double const *weight_in) {
    wjelem.resize(Nspecies);
    std::copy(weight_in, weight_in + Nspecies, wjelem.data());
}

void Bispectrum::compute_ui(int const jnum) {
    zero_uarraytot();
    addself_uarraytot(wself);

    double x, y, z;
    double r, rsq;
    double z0;
    double theta0;

    for (int j = 0; j < jnum; ++j) {
        // x = rij[j * 3 + 0];
        // y = rij[j * 3 + 1];
        // z = rij[j * 3 + 2];

        x = rij(j, 0);
        y = rij(j, 1);
        z = rij(j, 2);

        rsq = x * x + y * y + z * z;
        r = std::sqrt(rsq);

        // TODO this is not in agreement with the paper, maybe change it
        theta0 = (r - rmin0) * rfac0 * MY_PI / (rcutij[j] - rmin0);
        // theta0 = (r - rmin0) * rscale0;

        z0 = r / std::tan(theta0);
        compute_uarray(x, y, z, z0, r);
        add_uarraytot(r, wj[j], rcutij[j]);
    }
}

void Bispectrum::compute_zi() {
    for (int j1 = 0; j1 <= twojmax; ++j1) {
        for (int j2 = 0; j2 <= j1; ++j2) {
            for (int j = j1 - j2; j <= std::min(twojmax, j1 + j2); j += 2) {
                for (int mb = 0; 2 * mb <= j; ++mb) {
                    for (int ma = 0; ma <= j; ++ma) {
                        zarray_r(j1, j2, j, ma, mb) = 0.0;
                        zarray_i(j1, j2, j, ma, mb) = 0.0;

                        for (int ma1 = std::max(0, (2 * ma - j - j2 + j1) / 2);
                             ma1 <= std::min(j1, (2 * ma - j + j2 + j1) / 2);
                             ma1++) {
                            double sumb1_r = 0.0;
                            double sumb1_i = 0.0;

                            int const ma2 = (2 * ma - j - (2 * ma1 - j1) + j2) / 2;

                            for (int mb1 = std::max(0, (2 * mb - j - j2 + j1) / 2);
                                 mb1 <= std::min(j1, (2 * mb - j + j2 + j1) / 2);
                                 mb1++) {
                                int const mb2 = (2 * mb - j - (2 * mb1 - j1) + j2) / 2;

                                sumb1_r
                                        += cgarray(j1, j2, j, mb1, mb2)
                                           * (uarraytot_r(j1, ma1, mb1) * uarraytot_r(j2, ma2, mb2)
                                              - uarraytot_i(j1, ma1, mb1)
                                                * uarraytot_i(j2, ma2, mb2));
                                sumb1_i
                                        += cgarray(j1, j2, j, mb1, mb2)
                                           * (uarraytot_r(j1, ma1, mb1) * uarraytot_i(j2, ma2, mb2)
                                              + uarraytot_i(j1, ma1, mb1)
                                                * uarraytot_r(j2, ma2, mb2));
                            }

                            zarray_r(j1, j2, j, ma, mb)
                                    += sumb1_r * cgarray(j1, j2, j, ma1, ma2);
                            zarray_i(j1, j2, j, ma, mb)
                                    += sumb1_i * cgarray(j1, j2, j, ma1, ma2);
                        }
                    }
                }
            }
        }
    }
}

void Bispectrum::compute_bi() {
    for (int j1 = 0; j1 <= twojmax; ++j1) {
        for (int j2 = 0; j2 <= j1; ++j2) {
            for (int j = std::abs(j1 - j2); j <= std::min(twojmax, j1 + j2); j += 2) {
                barray(j1, j2, j) = 0.0;

                for (int mb = 0; 2 * mb < j; ++mb) {
                    for (int ma = 0; ma <= j; ++ma) {
                        barray(j1, j2, j)
                                += uarraytot_r(j, ma, mb) * zarray_r(j1, j2, j, ma, mb)
                                   + uarraytot_i(j, ma, mb) * zarray_i(j1, j2, j, ma, mb);
                    }
                }

                // For j even, special treatment for middle column
                if (j % 2 == 0) {
                    int const mb = j / 2;
                    for (int ma = 0; ma < mb; ++ma) {
                        barray(j1, j2, j)
                                += uarraytot_r(j, ma, mb) * zarray_r(j1, j2, j, ma, mb)
                                   + uarraytot_i(j, ma, mb) * zarray_i(j1, j2, j, ma, mb);
                    }

                    // ma = mb
                    barray(j1, j2, j)
                            += (uarraytot_r(j, mb, mb) * zarray_r(j1, j2, j, mb, mb)
                                + uarraytot_i(j, mb, mb) * zarray_i(j1, j2, j, mb, mb))
                               * 0.5;
                }

                barray(j1, j2, j) *= 2.0;

                if (bzero_flag) { barray(j1, j2, j) -= bzero[j]; }
            }
        }
    }
}

void Bispectrum::copy_bi2bvec() {
    switch (diagonalstyle) {
        case (0): {
            for (int j1 = 0, ncount = 0; j1 <= twojmax; ++j1) {
                for (int j2 = 0; j2 <= j1; ++j2) {
                    for (int j = std::abs(j1 - j2); j <= std::min(twojmax, j1 + j2);
                         j += 2) {
                        bvec[ncount++] = barray(j1, j2, j);
                    }
                }
            }
            return;
        }
        case (1): {
            for (int j1 = 0, ncount = 0; j1 <= twojmax; ++j1) {
                for (int j = 0; j <= std::min(twojmax, 2 * j1); j += 2) {
                    bvec[ncount++] = barray(j1, j1, j);
                }
            }
            return;
        }
        case (2): {
            for (int j1 = 0, ncount = 0; j1 <= twojmax; ++j1) {
                bvec[ncount++] = barray(j1, j1, j1);
            }
            return;
        }
        case (3): {
            for (int j1 = 0, ncount = 0; j1 <= twojmax; ++j1) {
                for (int j2 = 0; j2 <= j1; ++j2) {
                    for (int j = std::abs(j1 - j2); j <= std::min(twojmax, j1 + j2);
                         j += 2) {
                        if (j >= j1) { bvec[ncount++] = barray(j1, j2, j); }
                    }
                }
            }
            return;
        }
    }
}

void Bispectrum::zero_uarraytot() {
    for (int j = 0; j <= twojmax; ++j) {
        for (int ma = 0; ma <= j; ++ma) {
            for (int mb = 0; mb <= j; ++mb) {
                uarraytot_r(j, ma, mb) = 0.0;
                uarraytot_i(j, ma, mb) = 0.0;
            }
        }
    }
}

void Bispectrum::addself_uarraytot(double const wself_in) {
    for (int j = 0; j <= twojmax; ++j) {
        for (int ma = 0; ma <= j; ++ma) {
            uarraytot_r(j, ma, ma) = wself_in;
            uarraytot_i(j, ma, ma) = 0.0;
        }
    }
}

void Bispectrum::compute_uarray(double const x,
                                double const y,
                                double const z,
                                double const z0,
                                double const r) {
    // compute Cayley-Klein parameters for unit quaternion
    double const r0inv = 1.0 / std::sqrt(r * r + z0 * z0);

    double const a_r = r0inv * z0;
    double const a_i = -r0inv * z;
    double const b_r = r0inv * y;
    double const b_i = -r0inv * x;

    double rootpq;

    // VMK Section 4.8.2
    uarray_r(0, 0, 0) = 1.0;
    uarray_i(0, 0, 0) = 0.0;

    for (int j = 1; j <= twojmax; ++j) {
        // fill in left side of matrix layer from previous layer
        for (int mb = 0; 2 * mb <= j; ++mb) {
            uarray_r(j, 0, mb) = 0.0;
            uarray_i(j, 0, mb) = 0.0;

            for (int ma = 0; ma < j; ++ma) {
                rootpq = rootpqarray[(j - ma) * twojmax + (j - mb)];
                uarray_r(j, ma, mb) += rootpq
                                       * (a_r * uarray_r(j - 1, ma, mb)
                                          + a_i * uarray_i(j - 1, ma, mb));
                uarray_i(j, ma, mb) += rootpq
                                       * (a_r * uarray_i(j - 1, ma, mb)
                                          - a_i * uarray_r(j - 1, ma, mb));

                rootpq = rootpqarray[(ma + 1) * twojmax + (j - mb)];
                uarray_r(j, ma + 1, mb)
                        = -rootpq
                          * (b_r * uarray_r(j - 1, ma, mb) + b_i * uarray_i(j - 1, ma, mb));
                uarray_i(j, ma + 1, mb)
                        = -rootpq
                          * (b_r * uarray_i(j - 1, ma, mb) - b_i * uarray_r(j - 1, ma, mb));
            }
        }

        // copy left side to right side with inversion symmetry VMK 4.4(2)
        // u[ma-j, mb-j] = (-1)^(ma-mb)*Conj([u[ma, mb])

        int mbpar = -1;
        for (int mb = 0; 2 * mb <= j; ++mb) {
            mbpar = -mbpar;

            int mapar = -mbpar;
            for (int ma = 0; ma <= j; ++ma) {
                mapar = -mapar;
                if (mapar == 1) {
                    uarray_r(j, j - ma, j - mb) = uarray_r(j, ma, mb);
                    uarray_i(j, j - ma, j - mb) = -uarray_i(j, ma, mb);
                } else {
                    uarray_r(j, j - ma, j - mb) = -uarray_r(j, ma, mb);
                    uarray_i(j, j - ma, j - mb) = uarray_i(j, ma, mb);
                }
            }
        }
    }
}

void Bispectrum::add_uarraytot(double const r,
                               double const wj_in,
                               double const rcut_in) {
    double sfac = compute_sfac(r, rcut_in);
    sfac *= wj_in;

    for (int j = 0; j <= twojmax; ++j) {
        for (int ma = 0; ma <= j; ++ma) {
            for (int mb = 0; mb <= j; ++mb) {
                uarraytot_r(j, ma, mb) += sfac * uarray_r(j, ma, mb);
                uarraytot_i(j, ma, mb) += sfac * uarray_i(j, ma, mb);
            }
        }
    }
}

double Bispectrum::compute_sfac(double const r, double const rcut_in) {
    switch (switch_flag) {
        case (0):
            return 1.0;
        case (1):
            return (r <= rmin0) ? 1.0
                                : (r > rcut_in)
                                  ? 0.0
                                  : 0.5
                                    * (std::cos((r - rmin0) * MY_PI / (rcut_in - rmin0))
                                       + 1.0);
        default:
            return 0.0;
    }
}

double Bispectrum::compute_dsfac(double const r, double const rcut_in) {
    switch (switch_flag) {
        case (1):
            if (r <= rmin0 || r > rcut_in) { return 0.0; }
            else {
                double const rcutfac = MY_PI / (rcut_in - rmin0);
                return -0.5 * std::sin((r - rmin0) * rcutfac) * rcutfac;
            }
        default:
            return 0.0;
    }
}

void Bispectrum::create_twojmax_arrays() {
    int const jdim = twojmax + 1;

    cgarray.resize(jdim, jdim, jdim, jdim, jdim);
    rootpqarray.resize((jdim + 1) * (jdim + 1));
    barray.resize(jdim, jdim, jdim);
    dbarray.resize(jdim, jdim, jdim, 3);
    duarray_r.resize(jdim, jdim, jdim, 3);
    duarray_i.resize(jdim, jdim, jdim, 3);
    uarray_r.resize(jdim, jdim, jdim);
    uarray_i.resize(jdim, jdim, jdim);

    if (bzero_flag) { bzero.resize(jdim, static_cast<double>(0)); }

    if (!use_shared_arrays) {
        uarraytot_r.resize(jdim, jdim, jdim);
        uarraytot_i.resize(jdim, jdim, jdim);

        zarray_r.resize(jdim, jdim, jdim, jdim, jdim);
        zarray_i.resize(jdim, jdim, jdim, jdim, jdim);
    }
}

inline double Bispectrum::factorial(int const n) {
    if (n < 0 || n > nmaxfactorial) {
        std::cerr << "The input n = " + std::to_string(n)
                     + " is not valid for a factorial!";
        std::abort();
    }
    return nfac_table[n];
}

double const Bispectrum::nfac_table[] = {
        1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880,
        3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000,
        20922789888000, 355687428096000, 6.402373705728e+15, 1.21645100408832e+17,
        2.43290200817664e+18, 5.10909421717094e+19, 1.12400072777761e+21, 2.5852016738885e+22,
        6.20448401733239e+23, 1.5511210043331e+25, 4.03291461126606e+26, 1.08888694504184e+28,
        3.04888344611714e+29, 8.8417619937397e+30, 2.65252859812191e+32, 8.22283865417792e+33,
        2.63130836933694e+35, 8.68331761881189e+36, 2.95232799039604e+38, 1.03331479663861e+40,
        3.71993326789901e+41, 1.37637530912263e+43, 5.23022617466601e+44, 2.03978820811974e+46,
        8.15915283247898e+47,3.34525266131638e+49, 1.40500611775288e+51, 6.04152630633738e+52,
        2.65827157478845e+54, 1.1962222086548e+56,5.50262215981209e+57, 2.58623241511168e+59,
        1.24139155925361e+61, 6.08281864034268e+62, 3.04140932017134e+64,1.55111875328738e+66,
        8.06581751709439e+67, 4.27488328406003e+69, 2.30843697339241e+71, 1.26964033536583e+73,
        7.10998587804863e+74, 4.05269195048772e+76, 2.35056133128288e+78, 1.3868311854569e+80,
        8.32098711274139e+81,5.07580213877225e+83, 3.14699732603879e+85, 1.98260831540444e+87,
        1.26886932185884e+89, 8.24765059208247e+90,5.44344939077443e+92, 3.64711109181887e+94,
        2.48003554243683e+96, 1.71122452428141e+98, 1.19785716699699e+100,8.50478588567862e+101,
        6.12344583768861e+103, 4.47011546151268e+105, 3.30788544151939e+107,2.48091408113954e+109,
        1.88549470166605e+111, 1.45183092028286e+113, 1.13242811782063e+115,8.94618213078297e+116,
        7.15694570462638e+118, 5.79712602074737e+120, 4.75364333701284e+122,
        3.94552396972066e+124, 3.31424013456535e+126, 2.81710411438055e+128, 2.42270953836727e+130,
        2.10775729837953e+132, 1.85482642257398e+134, 1.65079551609085e+136, 1.48571596448176e+138,
        1.3520015276784e+140, 1.24384140546413e+142, 1.15677250708164e+144, 1.08736615665674e+146,
        1.03299784882391e+148, 9.91677934870949e+149, 9.61927596824821e+151, 9.42689044888324e+153,
        9.33262154439441e+155, 9.33262154439441e+157, 9.42594775983835e+159, 9.61446671503512e+161,
        9.90290071648618e+163, 1.02990167451456e+166, 1.08139675824029e+168, 1.14628056373471e+170,
        1.22652020319614e+172, 1.32464181945183e+174, 1.44385958320249e+176, 1.58824554152274e+178,
        1.76295255109024e+180, 1.97450685722107e+182, 2.23119274865981e+184, 2.54355973347219e+186,
        2.92509369349301e+188, 3.3931086844519e+190, 3.96993716080872e+192, 4.68452584975429e+194,
        5.5745857612076e+196,6.68950291344912e+198, 8.09429852527344e+200, 9.8750442008336e+202,
        1.21463043670253e+205,1.50614174151114e+207, 1.88267717688893e+209, 2.37217324288005e+211,
        3.01266001845766e+213,3.8562048236258e+215, 4.97450422247729e+217, 6.46685548922047e+219,
        8.47158069087882e+221,1.118248651196e+224,1.48727070609069e+226, 1.99294274616152e+228,
        2.69047270731805e+230, 3.65904288195255e+232,5.01288874827499e+234, 6.91778647261949e+236,
        9.61572319694109e+238, 1.34620124757175e+241,
        1.89814375907617e+243, 2.69536413788816e+245, 3.85437071718007e+247, 5.5502938327393e+249,
        8.04792605747199e+251, 1.17499720439091e+254, 1.72724589045464e+256, 2.55632391787286e+258,
        3.80892263763057e+260, 5.71338395644585e+262, 8.62720977423323e+264, 1.31133588568345e+267,
        2.00634390509568e+269, 3.08976961384735e+271, 4.78914290146339e+273, 7.47106292628289e+275,
        1.17295687942641e+278, 1.85327186949373e+280, 2.94670227249504e+282, 4.71472363599206e+284,
        7.59070505394721e+286, 1.22969421873945e+289, 2.0044015765453e+291, 3.28721858553429e+293,
        5.42391066613159e+295, 9.00369170577843e+297, 1.503616514865e+300,  // nmaxfactorial = 167
};

inline double Bispectrum::deltacg(int const j1, int const j2, int const j) {
    double const sfaccg = factorial((j1 + j2 + j) / 2 + 1);
    return std::sqrt(factorial((j1 + j2 - j) / 2) * factorial((j1 - j2 + j) / 2)
                     * factorial((-j1 + j2 + j) / 2) / sfaccg);
}

void Bispectrum::init_clebsch_gordan() {
    for (int j1 = 0; j1 <= twojmax; ++j1) {
        for (int j2 = 0; j2 <= twojmax; ++j2) {
            for (int j = std::abs(j1 - j2); j <= std::min(twojmax, j1 + j2); j += 2) {
                for (int m1 = 0; m1 <= j1; m1 += 1) {
                    int const aa2 = 2 * m1 - j1;

                    for (int m2 = 0; m2 <= j2; m2 += 1) {
                        // -c <= cc <= c
                        int const bb2 = 2 * m2 - j2;
                        int const m = (aa2 + bb2 + j) / 2;
                        if (m < 0 || m > j) { continue; }

                        double sum(0.0);
                        for (int z = std::max(
                                0, std::max(-(j - j2 + aa2) / 2, -(j - j1 - bb2) / 2));
                             z <= std::min((j1 + j2 - j) / 2,
                                           std::min((j1 - aa2) / 2, (j2 + bb2) / 2));
                             z++) {
                            int const ifac = z % 2 ? -1 : 1;
                            sum += ifac
                                   / (factorial(z) * factorial((j1 + j2 - j) / 2 - z)
                                      * factorial((j1 - aa2) / 2 - z)
                                      * factorial((j2 + bb2) / 2 - z)
                                      * factorial((j - j2 + aa2) / 2 + z)
                                      * factorial((j - j1 - bb2) / 2 + z));
                        }

                        int const cc2 = 2 * m - j;

                        double dcg = deltacg(j1, j2, j);

                        double sfaccg = std::sqrt(
                                factorial((j1 + aa2) / 2) * factorial((j1 - aa2) / 2)
                                * factorial((j2 + bb2) / 2) * factorial((j2 - bb2) / 2)
                                * factorial((j + cc2) / 2) * factorial((j - cc2) / 2)
                                * (j + 1));

                        cgarray(j1, j2, j, m1, m2) = sum * dcg * sfaccg;
                    }
                }
            }
        }
    }
}

void Bispectrum::init_rootpqarray() {
    for (int p = 1; p <= twojmax; p++) {
        for (int q = 1; q <= twojmax; q++) {
            rootpqarray[p * twojmax + q] = std::sqrt(static_cast<double>(p) / q);
        }
    }
}

inline void Bispectrum::jtostr(char *str_out, int const j) {
    if (j % 2 == 0) { sprintf(str_out, "%d", j / 2); }
    else {
        sprintf(str_out, "%d/2", j);
    }
}

inline void Bispectrum::mtostr(char *str_out, int const j, int const m) {
    if (j % 2 == 0) { sprintf(str_out, "%d", m - j / 2); }
    else {
        sprintf(str_out, "%d/2", 2 * m - j);
    }
}

void Bispectrum::print_clebsch_gordan(FILE *file) {
    char stra[20];
    char strb[20];
    char strc[20];
    char straa[20];
    char strbb[20];
    char strcc[20];

    int m, aa2, bb2;

    fprintf(file, "a, aa, b, bb, c, cc, c(a,aa,b,bb,c,cc) \n");

    for (int j1 = 0; j1 <= twojmax; ++j1) {
        jtostr(stra, j1);

        for (int j2 = 0; j2 <= twojmax; ++j2) {
            jtostr(strb, j2);

            for (int j = std::abs(j1 - j2); j <= std::min(twojmax, j1 + j2); j += 2) {
                jtostr(strc, j);

                for (int m1 = 0; m1 <= j1; ++m1) {
                    mtostr(straa, j1, m1);

                    aa2 = 2 * m1 - j1;

                    for (int m2 = 0; m2 <= j2; ++m2) {
                        bb2 = 2 * m2 - j2;
                        m = (aa2 + bb2 + j) / 2;

                        if (m < 0 || m > j) { continue; }

                        mtostr(strbb, j2, m2);

                        mtostr(strcc, j, m);

                        fprintf(file,
                                "%s\t%s\t%s\t%s\t%s\t%s\t%g\n",
                                stra,
                                straa,
                                strb,
                                strbb,
                                strc,
                                strcc,
                                cgarray(j1, j2, j, m1, m2));
                    }
                }
            }
        }
    }
}

int Bispectrum::compute_ncoeff() {
    switch (diagonalstyle) {
        case (0): {
            int ncount(0);
            for (int j1 = 0; j1 <= twojmax; ++j1) {
                for (int j2 = 0; j2 <= j1; ++j2) {
                    for (int j = std::abs(j1 - j2); j <= std::min(twojmax, j1 + j2);
                         j += 2) {
                        ncount++;
                    }
                }
            }
            return ncount;
        }
        case (1): {
            int ncount(0);
            for (int j1 = 0; j1 <= twojmax; ++j1) {
                for (int j = 0; j <= std::min(twojmax, 2 * j1); j += 2) { ncount++; }
            }
            return ncount;
        }
        case (2): {
            int ncount(0);
            for (int j1 = 0; j1 <= twojmax; ++j1) { ncount++; }
            return ncount;
        }
        case (3): {
            int ncount(0);
            for (int j1 = 0; j1 <= twojmax; ++j1) {
                for (int j2 = 0; j2 <= j1; ++j2) {
                    for (int j = std::abs(j1 - j2); j <= std::min(twojmax, j1 + j2);
                         j += 2) {
                        if (j >= j1) { ncount++; }
                    }
                }
            }
            return ncount;
        }
        default:
            std::cerr << "The input style index = " + std::to_string(diagonalstyle)
                         + " is not a valid index!!";
            std::abort();
    }
}

Bispectrum::Bispectrum(std::string &file_name) {
    initFromFile(file_name);
}

void Bispectrum::initFromFile(std::string &file_name) {
    std::fstream file_ptr(file_name);
//    std::string placeholder_string;
    std::string line;
    std::vector<int> int_params;
    std::vector<double> double_params;
    std::vector<std::string> string_params;


    std::string cutoff_type;
    int descriptor_size, jmax, diagonalstyle_, switchflag, bzeroflag;
    std::vector<std::string> species;
    std::map<std::pair<std::string, std::string>, double> species_pair_cutoffs;
    bool normalize = false;
    std::vector<double> weights;
    double cutoff, rfac0_, rmin0_;

    std::ifstream file = FileIOUtils::open_file(file_name);

    // read Cutoff type name
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_string_params(line, string_params, 1);
    cutoff_type = string_params[0];
    string_params.clear();
    line.clear();

    // read number of species
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_int_params(line, int_params, 1);
    n_species = int_params[0];
    int_params.clear();
    line.clear();

        // read species names and cutoffs
    std::set<std::string> species_set;
    std::pair<std::string, std::string> species_species_pair;
    int species_index = n_species;
    do {
        FileIOUtils::get_next_data_line(file, line);
        FileIOUtils::parse_string_params(line, string_params, 2);
        FileIOUtils::parse_double_params(line, double_params, 1);

        species_set.insert(string_params[0]);
        species_set.insert(string_params[1]);

        species_species_pair = std::make_pair(string_params[0], string_params[1]);
        species_pair_cutoffs[species_species_pair] = double_params[0];

        species_species_pair = std::make_pair(string_params[1], string_params[0]);
        species_pair_cutoffs[species_species_pair] = double_params[0];

        double_params.clear();
        string_params.clear();
        line.clear();
    } while (--species_index > 0);
    species = std::vector<std::string>(species_set.begin(), species_set.end());

    // get jmax
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_int_params(line, int_params, 1);
    jmax = int_params[0];
    twojmax = 2 * jmax;
    int_params.clear();
    line.clear();

    // rfac0
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_double_params(line, double_params, 1);
    rfac0 = double_params[0];
    double_params.clear();
    line.clear();

    // diagonalstyle
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_int_params(line, int_params, 1);
    diagonalstyle = int_params[0];
    int_params.clear();
    line.clear();

    // rmin0
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_double_params(line, double_params, 1);
    rmin0 = double_params[0];
    double_params.clear();
    line.clear();

    // switchflag
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_int_params(line, int_params, 1);
    switch_flag = int_params[0];
    int_params.clear();
    line.clear();

    // bzeroflag
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_int_params(line, int_params, 1);
    bzero_flag = int_params[0];
    int_params.clear();
    line.clear();

    // weights
    // read weights for each species
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_double_params(line, double_params, species.size());
    weights = double_params;
    double_params.clear();
    line.clear();

    // Initialize the descriptor
    auto cutoff_matrix = std::make_unique<double[]>(n_species * n_species);
    for (int i = 0; i < n_species; i++) {
        for (int j = 0; j < n_species; j++) {
            species_species_pair = std::make_pair(species[i], species[j]);
            cutoff_matrix[i * n_species + j] = species_pair_cutoffs[species_species_pair];
        }
    }

    set_cutoff(cutoff_type.c_str(), n_species, cutoff_matrix.get());
    set_weight(n_species, weights.data());
    set_species(n_species);

    use_shared_arrays = 0;
    // This value is maximum number of neighbors possible,
    // Ideally it should be determined at runtime, but enzyme has issues with it
    // so fixing the upperbound. Would be made an hyper-parameter soon, till
    // Enzyme people fixes it. TODO
    nmax = 0;
    int max_nmax = 25;
    grow_rij(max_nmax);

    twojmax = 2 * jmax;
    wself = 1.0;

    ncoeff = compute_ncoeff();

    create_twojmax_arrays();

    if (bzero_flag) {
        double const www = wself * wself * wself;
        for (int j = 1; j <= twojmax + 1; ++j) { bzero[j] = www * j; }
    }

    // 1D
    bvec.resize(ncoeff, static_cast<double>(0));

    // 2D
    dbvec.resize(ncoeff, 3);

    build_indexlist();

    init();

    width = get_width();


}

void Bispectrum::clone_empty(DescriptorKind *descriptorKind) {
    auto d_bs = dynamic_cast<Bispectrum *>(descriptorKind);
    twojmax = d_bs->twojmax;
    switch_flag = d_bs->switch_flag;
    bzero_flag = d_bs->bzero_flag;
    diagonalstyle = d_bs->diagonalstyle;
    width = d_bs->width;
    wself = d_bs->wself;
    use_shared_arrays = d_bs->use_shared_arrays;
    n_species = d_bs->n_species;

    // To be diffed against
    rmin0 = 0.0;
    rfac0 = 0.0;
    nmax = 0;
    int max_nmax = 250;
    grow_rij(max_nmax);

    auto weights = new double[n_species];
    auto cutoff_matrix = new double[n_species * n_species];
    for (int i = 0; i < n_species; i++) {
        weights[i] = 0;
        for (int j = 0; j < n_species; j++) {
            cutoff_matrix[i * n_species + j] = 0.0;
        }
    }

    //TODO initialize from d_bs
    std::string cutoff_function = "cos";

    set_weight(n_species, weights);
    set_cutoff(cutoff_function.c_str(), n_species, cutoff_matrix);
    ncoeff = compute_ncoeff();
    create_twojmax_arrays();
    if (bzero_flag) {
        double const www = wself * wself * wself;
        for (int j = 1; j <= twojmax + 1; ++j) { bzero[j] = www * j; }
    }
    bvec.resize(ncoeff, static_cast<double>(0));
    dbvec.resize(ncoeff, 3);
    build_indexlist();
    init();

    delete[] cutoff_matrix;
    delete[] weights;
}


int Bispectrum::get_width() {
    int N = 0;
    for (int j1 = 0; j1 <= twojmax; j1++) {
        if (diagonalstyle == 2) {
            N += 1;
        } else if (diagonalstyle == 1) {
            for (int j = 0; j <= std::min(twojmax, 2 * j1); j += 2) {
                N += 1;
            }
        } else if (diagonalstyle == 0) {
            for (int j2 = 0; j2 <= j1; j2++) {
                for (int j = j1 - j2; j <= std::min(twojmax, j1 + j2); j += 2) {
                    N += 1;
                }
            }
        } else if (diagonalstyle == 3) {
            for (int j2 = 0; j2 <= j1; j2++) {
                for (int j = j1 - j2; j <= std::min(twojmax, j1 + j2); j += 2) {
                    if (j >= j1) {
                        N += 1;
                    }
                }
            }
        }
    }
    return N;
}

// helper.cpp

std::string
FormatMessageFileLineFunctionMessage(std::string const & message1,
                                     std::string const & fileName,
                                     long lineNumber,
                                     std::string const & functionName,
                                     std::string const & message2)
{
  std::ostringstream ss;
  ss << "\n";
  ss << message1 << ":" << fileName << ":" << lineNumber << ":@("
     << functionName << ")\n";
  ss << message2 << "\n\n";
  return ss.str();
}
template<>
_Array_Basic<std::string>::_Array_Basic(std::size_t const count) :
    m(count, "\0")
{
}

template<>
inline void Array2D<std::string>::resize(int const extentZero,
                                         int const extentOne)
{
  _extentZero = extentZero;
  _extentOne = extentOne;
  std::size_t const _n = _extentZero * _extentOne;
  this->m.resize(_n, "\0");
}

template<>
inline void Array3D<std::string>::resize(int const extentZero,
                                         int const extentOne,
                                         int const extentTwo)
{
  _extentZero = extentZero;
  _extentOne = extentOne;
  _extentTwo = extentTwo;
  std::size_t const _n = _extentZero * _extentOne * _extentTwo;
  this->m.resize(_n, "\0");
}

template<>
inline void Array4D<std::string>::resize(int const extentZero,
                                         int const extentOne,
                                         int const extentTwo,
                                         int const extentThree)
{
  _extentZero = extentZero;
  _extentOne = extentOne;
  _extentTwo = extentTwo;
  _extentThree = extentThree;
  std::size_t const _n = _extentZero * _extentOne * _extentTwo * _extentThree;
  this->m.resize(_n, "\0");
}

template<>
inline void Array5D<std::string>::resize(int const extentZero,
                                         int const extentOne,
                                         int const extentTwo,
                                         int const extentThree,
                                         int const extentFour)
{
  _extentZero = extentZero;
  _extentOne = extentOne;
  _extentTwo = extentTwo;
  _extentThree = extentThree;
  _extentFour = extentFour;
  std::size_t const _n
      = _extentZero * _extentOne * _extentTwo * _extentThree * _extentFour;
  this->m.resize(_n, "\0");
}

// SOAP.cpp
#define MAX_NEIGHBORS 100

using namespace Descriptor;

SOAP::SOAP(int n_max, int l_max, double cutoff, std::vector<std::string> &species, std::string radial_basis,
           double eta) {
    this->n_max = n_max;
    this->l_max = l_max;
    this->cutoff = cutoff;
    this->species_ = species;
    this->n_species = species.size();
    this->radial_basis = std::move(radial_basis);
    this->eta = eta;
    this->l_max_sq = (l_max + 1) * (l_max + 1);
    allocate_memory();
    init_radial_basis_array();
    this->width = get_width();
}

int SOAP::get_width() {
    if (width == -1) {
        width = (((n_species + 1) * n_species) / 2 * (n_max * (n_max + 1)) * (l_max + 1)) / 2;
    }
    return this->width;
}

void SOAP::init_radial_basis_array() {
    if (radial_basis == "polynomial") {
        gl_quad_weights = get_gl_weights();
        gl_quad_radial_grid_points = get_gl_grid(cutoff);
        n_gl_quad_points = gl_quad_weights.size();

        radial_basis_array = std::vector<double>(n_max * n_gl_quad_points);
        polynomial_basis(n_max, cutoff, n_gl_quad_points, gl_quad_radial_grid_points.data(),
                         radial_basis_array.data());
        for (int i = 0; i < n_gl_quad_points; i++) {
            gl_quad_radial_sq_grid_points[i] = gl_quad_radial_grid_points[i] * gl_quad_radial_grid_points[i];
            exp_eta_r2[i] = std::exp(-1 * eta * gl_quad_radial_sq_grid_points[i]);
        }
    } else {
        throw std::invalid_argument("radial_basis must be one of: polynomial");
    }
}

void
SOAP::compute(int index, int n_atoms, int *species, int *neighbor_lists, int number_of_neighbors, double *coordinates,
              double *desc) {
    auto *coordinates_ = (VectorOfSizeDIM *) coordinates;
    std::array<double, 3> r_i = {coordinates_[index][0], coordinates_[index][1], coordinates_[index][2]};

    // TODO: these 2 coord allocations are needed in every call, else Enzyme fails/gives different results
    // Restest them with later versions of Enzyme
    std::fill(I_zj_real.begin(), I_zj_real.end(), 0.0);
    std::fill(I_zj_imag.begin(), I_zj_imag.end(), 0.0);
    auto center_shifted_neighbor_coordinates = std::vector<double>(3 * MAX_NEIGHBORS);
    auto center_shifted_neighbor_coordinates_zj = std::vector<double>(3 * MAX_NEIGHBORS);


    //1. get center shifted coordinates for index atom. Neighors exclude the center atom
    auto i_coordinates = (VectorOfSizeDIM *) center_shifted_neighbor_coordinates.data();
    for (int i = 0; i < number_of_neighbors; i++) {
        int neighbor_index = neighbor_lists[i];
        std::array<double, 3> r_j = {coordinates_[neighbor_index][0], coordinates_[neighbor_index][1],
                                     coordinates_[neighbor_index][2]};
        i_coordinates[i][0] = r_j[0] - r_i[0];
        i_coordinates[i][1] = r_j[1] - r_i[1];
        i_coordinates[i][2] = r_j[2] - r_i[2];
    }
    //2. get all the neighbor coordinates of a given species
    for (int zj = 0; zj < n_species; zj++) {
        int n_neighbors_zj = 0;
        std::vector<int> species_j_indices(MAX_NEIGHBORS, -1);
        int i_neigh_j = 0;
        for (int i = 0; i < number_of_neighbors; i++) {
            if (species[neighbor_lists[i]] == zj) {
                species_j_indices[i_neigh_j] = i;
                n_neighbors_zj++;
                i_neigh_j++;
            }
        }
        auto j_coordinates = (VectorOfSizeDIM *) center_shifted_neighbor_coordinates_zj.data();
        for (int j = 0; j < n_neighbors_zj; j++) {
            j_coordinates[j][0] = i_coordinates[species_j_indices[j]][0];
            j_coordinates[j][1] = i_coordinates[species_j_indices[j]][1];
            j_coordinates[j][2] = i_coordinates[species_j_indices[j]][2];
        }

        //3. For each species element, get Ylmi and store
        //4. For each species element, get I_ij and store
        std::fill(Ylmi_real.begin(), Ylmi_real.begin() + l_max_sq * n_neighbors_zj, 0.0);
        std::fill(Ylmi_imag.begin(), Ylmi_imag.begin() + l_max_sq * n_neighbors_zj, 0.0);
        for (int j = 0; j < n_neighbors_zj; j++) {
            std::array<double, 3> r_j = {j_coordinates[j][0], j_coordinates[j][1], j_coordinates[j][2]};

            Ylmi_all_l_from_r(l_max, r_j.data(),
                              Ylmi_real.data() + j * l_max_sq,
                              Ylmi_imag.data() + j * l_max_sq);

            double r_ij_sq = r_j[0] * r_j[0] + r_j[1] * r_j[1] + r_j[2] * r_j[2];
            double r_ij = std::sqrt(r_ij_sq);

            //5. Sum over all the I_ij of a given species element
            double two_eta_r2_rij = 0;
            int lm_idx = -1;
            for (int l = 0; l < l_max + 1; l++) {
                for (int m = -l; m <= l; m++) {
                    lm_idx = l * l + l + m;
                    for (int i = 0; i < n_gl_quad_points; i++) {
                        int idx = n_gl_quad_points * lm_idx + i;
                        two_eta_r2_rij = spherical_in(l, 2 * eta * gl_quad_radial_grid_points[i] * r_ij) *
                                         std::exp(-1 * eta * r_ij_sq) * exp_eta_r2[i];
                        I_zj_real[idx] += two_eta_r2_rij * Ylmi_real[j * l_max_sq + lm_idx];
                        I_zj_imag[idx] += two_eta_r2_rij * Ylmi_imag[j * l_max_sq + lm_idx];
                    }
                }
            }
        }

        //6. numerically integrate over the radial grid points
        //Cij_real = \int r^2 R(r) * I_zj_real, Cij_imag = \int r^2 R(r) * I_zj_imag = \sum_i r[i]^2 R(r[i]) * I_zj_imag[i] w_i
        std::fill(Cij_real.begin(), Cij_real.end(), 0.0);
        std::fill(Cij_imag.begin(), Cij_imag.end(), 0.0);
        int lm_idx = -1;
        double gl_weigth_r2 = 0;
        for (int n = 0; n < n_max; n++) {
            for (int l = 0; l < l_max + 1; l++) {
                for (int m = -l; m <= l; m++) {
                    lm_idx = l * l + l + m;
                    for (int i = 0; i < n_gl_quad_points; i++) {
                        gl_weigth_r2 = radial_basis_array[n * n_gl_quad_points + i] * gl_quad_radial_sq_grid_points[i] *
                                       gl_quad_weights[i];
                        Cij_real[n * l_max_sq + lm_idx] +=
                                gl_weigth_r2 * I_zj_real[l * (l + 1) * n_gl_quad_points + m * n_gl_quad_points + i];
                        Cij_imag[n * l_max_sq + lm_idx] +=
                                gl_weigth_r2 * I_zj_imag[l * (l + 1) * n_gl_quad_points + m * n_gl_quad_points + i];
                    }
                    //7. Store the Cznlm for each species element
                    Cznlm_real[zj * n_max * l_max_sq + n * l_max_sq + lm_idx] = Cij_real[n * l_max_sq + lm_idx];
                    Cznlm_imag[zj * n_max * l_max_sq + n * l_max_sq + lm_idx] = Cij_imag[n * l_max_sq + lm_idx];
                }
            }
        }
    }

    //8. compute the power spectrum for each species element
    int tmp_index = 0;//TODO: get proper index
    int lm_idx = -1;
    std::fill(power_spectrum.begin(), power_spectrum.end(), 0.0);
    for (int z_j = 0; z_j < n_species; z_j++) {
        for (int z_i = z_j; z_i < n_species; z_i++) {
            for (int n_j = 0; n_j < n_max; n_j++) {
                for (int n_i = n_j; n_i < n_max; n_i++) {
                    for (int l = 0; l < l_max + 1; l++) {
                        for (int m = -l; m <= l; m++) {
                            lm_idx = l * l + l + m;
                            power_spectrum[tmp_index] +=
                                    Cznlm_real[z_j * n_max * l_max_sq + n_j * l_max_sq + lm_idx] *
                                    Cznlm_real[z_i * n_max * l_max_sq + n_i * l_max_sq + lm_idx] +
                                    Cznlm_imag[z_j * n_max * l_max_sq + n_j * l_max_sq + lm_idx] *
                                    Cznlm_imag[z_i * n_max * l_max_sq + n_i * l_max_sq + lm_idx];
                        }
                        power_spectrum[tmp_index] *= 124.02510672119926 * std::sqrt(
                                8.0 / (2.0 * l + 1.0)); //  M_PI * 4 * std::pow(M_PI, 2) = 124.02510672119926
                        desc[tmp_index] = power_spectrum[tmp_index];
                        tmp_index++;
                    }
                }
            }
        }
    }
}


void SOAP::allocate_memory() {

    //1. spherical bessel I_ij functions (l_max + 1)**2 * n_gl_quad_points  (real and imaginary part)
    I_zj_real = std::vector<double>(l_max_sq * n_gl_quad_points); // preallocated
    I_zj_imag = std::vector<double>(l_max_sq * n_gl_quad_points); // preallocated

    //2. Projection Cnlm coefficients
    Cznlm_real = std::vector<double>(n_species * n_max * l_max_sq);
    Cznlm_imag = std::vector<double>(n_species * n_max * l_max_sq);
    Cij_real = std::vector<double>(n_max * l_max_sq);
    Cij_imag = std::vector<double>(n_max * l_max_sq);

    //3. spherical harmonics: first (l_max + 1)**2 is real part, second (l_max + 1)**2 is imaginary part
    Ylmi_real = std::vector<double>(l_max_sq * MAX_NEIGHBORS);
    Ylmi_imag = std::vector<double>(l_max_sq * MAX_NEIGHBORS);

    //4. exp(-eta * r**2) for each grid point for integration
    exp_eta_r2 = std::vector<double>(n_gl_quad_points);
    gl_quad_radial_sq_grid_points = std::vector<double>(n_gl_quad_points);

    // center_shifted_neighbor_coordinates = std::vector<double>(3 * MAX_NEIGHBORS);
    // center_shifted_neighbor_coordinates_zj = std::vector<double>(3 * MAX_NEIGHBORS);
    //5. power spectrum
    power_spectrum = std::vector<double>(n_species * (n_species + 1) / 2 * n_max * (n_max + 1) / 2 * (l_max + 1));

}

void SOAP::clone_empty(DescriptorKind *descriptorKind) {
    auto d_soap = dynamic_cast<SOAP *>(descriptorKind);
    n_max = d_soap->n_max;
    l_max = d_soap->l_max;
    cutoff = d_soap->cutoff;
    n_species = d_soap->n_species;
    eta = d_soap->eta;
    l_max_sq = d_soap->l_max_sq;
    allocate_memory();
    init_radial_basis_array();
    width = d_soap->width;
    // zero out the memory
    //    if (radial_basis == "polynomial") {
    //        for (int i = 0; i < d_soap->radial_basis_array.size(); i++) {
    //            radial_basis_array[i] = 0.0;
    //            gl_quad_radial_sq_grid_points[i] = 0.0;
    //            gl_quad_weights[i] = 0.0;
    //            gl_quad_radial_grid_points[i] = 0.0;
    //        }
    //    } else {
    //        throw std::invalid_argument("radial_basis must be one of: polynomial");
    //    }
}

SOAP::SOAP(std::string &filename) {

    // Open the descriptor file
    std::ifstream file = FileIOUtils::open_file(filename);

    // String containing data line and list of parameters
    std::vector<std::string> string_params;
    std::vector<double> double_params;
    std::vector<int> int_params;
    std::vector<bool> bool_params;
    std::string line;

    // params
    int n_max_, l_max_, n_species_;
    double cutoff_, eta_;
    std::vector<std::string> species;
    std::string radial_basis_;

    // File format:
    /*
     * # n_max
     * 2
     * # l_max
     * 2
     * # cutoff
     * 5.0
     * # eta
     * 1.0
     * # radial_basis
     * polynomial
     * # species
     * 4
     * H C N O
     */
    // n_max
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_int_params(line, int_params, 1);
    n_max_ = int_params[0];
    int_params.clear();
    line.clear();

    // l_max
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_int_params(line, int_params, 1);
    l_max_ = int_params[0];
    int_params.clear();
    line.clear();

    // cutoff
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_double_params(line, double_params, 1);
    cutoff_ = double_params[0];
    double_params.clear();
    line.clear();

    // eta
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_double_params(line, double_params, 1);
    eta_ = double_params[0];
    double_params.clear();
    line.clear();

    // radial_basis
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_string_params(line, string_params, 1);
    radial_basis_ = string_params[0];
    string_params.clear();
    line.clear();

    // species
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_int_params(line, int_params, 1);
    n_species_ = int_params[0];
    int_params.clear();
    line.clear();

    FileIOUtils::parse_string_params(line, string_params, n_species_);
    for (auto &species_name: string_params) {
        species.push_back(species_name);
    }
    string_params.clear();
    line.clear();

    file.close();

    // Initialize the descriptor
    this->n_max = n_max_;
    this->l_max = l_max_;
    this->cutoff = cutoff_;
    this->species_ = species;
    this->n_species = species.size();
    this->radial_basis = std::move(radial_basis);
    this->eta = eta_;
    l_max_sq = (l_max + 1) * (l_max + 1);
    init_radial_basis_array();
    allocate_memory();
    this->width = this->get_width();

}

// SymmetryFunction.cpp

inline void SymmetryFunctions::set_species(std::vector<std::string> &species) {
    species_.resize(species.size());
    std::copy(species.begin(), species.end(), species_.begin());
}

inline void SymmetryFunctions::get_species(std::vector<std::string> &species) {
    species.resize(species_.size());
    std::copy(species_.begin(), species_.end(), species.begin());
}

inline int SymmetryFunctions::get_num_species() { return species_.size(); }

void SymmetryFunctions::set_cutoff(char const *name,
                                   std::size_t const Nspecies,
                                   double const *rcut_2D) {
    (void) name;   // to avoid unused warning
    rcut_2D_.resize(Nspecies, Nspecies, rcut_2D);
}

inline double SymmetryFunctions::get_cutoff(int const iCode, int const jCode) {
    return rcut_2D_(iCode, jCode);
}

void SymmetryFunctions::add_descriptor(char const *name,
                                       double const *values,
                                       int const row,
                                       int const col) {
    if (strcmp(name, "g1") == 0) { name_.push_back(1); }
    if (strcmp(name, "g2") == 0) { name_.push_back(2); }
    if (strcmp(name, "g3") == 0) { name_.push_back(3); }
    if (strcmp(name, "g4") == 0) { name_.push_back(4); }
    if (strcmp(name, "g5") == 0) { name_.push_back(5); }

    Array2D<double> params(row, col, values);
    params_.push_back(std::move(params));

    auto sum = std::accumulate(num_param_sets_.begin(), num_param_sets_.end(), 0);
    starting_index_.push_back(sum);

    num_param_sets_.push_back(row);
    num_params_.push_back(col);

    if (strcmp(name, "g4") == 0 || strcmp(name, "g5") == 0) {
        has_three_body_ = true;
    }
}

inline double cut_cos(double const r, double const rcut) {
    return (r < rcut) ? 0.5 * (std::cos(MY_PI * r / rcut) + 1.0) : 0.0;
}

int SymmetryFunctions::get_num_descriptors() {
    return std::accumulate(num_param_sets_.begin(), num_param_sets_.end(), 0);
}

void sym_g5(double const zeta, double const lambda, double const eta, double const *r,
            double const *rcut, double &phi) {
    double const rij = r[0];
    double const rik = r[1];
    double const rcutij = rcut[0];
    double const rcutik = rcut[1];

    if (rij > rcutij || rik > rcutik) { phi = 0.0; }
    else {
        double const rjk = r[2];
        double const rijsq = rij * rij;
        double const riksq = rik * rik;
        double const rjksq = rjk * rjk;

        // index is the apex atom
        double const cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);
        double const base = 1.0 + lambda * cos_ijk;

        // prevent numerical instability (when lambda=-1 and cos_ijk=1)
        double const costerm = (base <= 0) ? 0.0 : std::pow(base, zeta);
        double const eterm = std::exp(-eta * (rijsq + riksq));
        phi = std::pow(2, 1 - zeta) * costerm * eterm * cut_cos(rij, rcutij)
              * cut_cos(rik, rcutik);
    }
}

void sym_g4(double const zeta, double const lambda, double const eta, double const *r,
            double const *rcut, double &phi) {
    double const rij = r[0];
    double const rik = r[1];
    double const rjk = r[2];
    double const rcutij = rcut[0];
    double const rcutik = rcut[1];
    double const rcutjk = rcut[2];

    if (rij > rcutij || rik > rcutik || rjk > rcutjk) { phi = 0.0; }
    else {
        double const rijsq = rij * rij;
        double const riksq = rik * rik;
        double const rjksq = rjk * rjk;

        // index is the apex atom
        double const cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);
        double const base = 1 + lambda * cos_ijk;

        // prevent numerical instability (when lambda=-1 and cos_ijk=1)
        double const costerm = (base <= 0) ? 0.0 : std::pow(base, zeta);
        double const eterm = std::exp(-eta * (rijsq + riksq + rjksq));

        phi = std::pow(2, 1 - zeta) * costerm * eterm * cut_cos(rij, rcutij)
              * cut_cos(rik, rcutik) * cut_cos(rjk, rcutjk);
    }
}

void sym_g3(double const kappa, double const r, double const rcut, double &phi) {
    phi = std::cos(kappa * r) * cut_cos(r, rcut);
}

void sym_g2(double const eta, double const Rs, double const r, double const rcut, double &phi) {
    phi = std::exp(-eta * (r - Rs) * (r - Rs)) * cut_cos(r, rcut);
}

void sym_g1(double const r, double const rcut, double &phi) {
    phi = cut_cos(r, rcut);
}

void SymmetryFunctions::compute(int const index,
                                int const n_atoms,
                                int *const species,
                                int *const neigh_list,
                                int const number_of_neigh,
                                double *const coords,
                                double *const desc) {
    // prepare data
    auto *coordinates = (VectorOfSizeDIM *) coords;
    int const iSpecies = species[index];
    // Setup loop over neighbors of current particle
    for (int jj = 0; jj < number_of_neigh; ++jj) {
        // adjust index of particle neighbor
        int const j = neigh_list[jj];
        int const jSpecies = species[j];
        // cutoff between ij
        double rcutij = rcut_2D_(iSpecies, jSpecies);
        // Compute rij
        double rij[DIM];
        for (int dim = 0; dim < DIM; ++dim) {
            rij[dim] = coordinates[j][dim] - coordinates[index][dim];
        }

        double const rijsq = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
        double const rijmag = std::sqrt(rijsq);

        // if particles index and j not interact
        if (rijmag > rcutij) { continue; }

        // Loop over descriptors
        // two-body descriptors
        for (std::size_t p = 0; p < name_.size(); ++p) {
            if (name_[p] != 1 && name_[p] != 2 && name_[p] != 3) {
                continue;
            }

            int idx = starting_index_[p];
            // Loop over same descriptor but different parameter set
            for (int q = 0; q < num_param_sets_[p]; ++q) {
                double gc = 0.0;

                if (name_[p] == 1) {
                    sym_g1(rijmag, rcutij, gc);
                } else if (name_[p] == 2) {
                    double eta = params_[p](q, 0);
                    auto Rs = params_[p](q, 1);
                    sym_g2(eta, Rs, rijmag, rcutij, gc);
                } else if (name_[p] == 3) {
                    double kappa = params_[p](q, 0);
                    sym_g3(kappa, rijmag, rcutij, gc);
                }
                desc[idx] += gc;
                ++idx;
            }
        }
        // three-body descriptors
        if (has_three_body_ == 0) { continue; }

        // Loop over kk
        for (int kk = jj + 1; kk < number_of_neigh; ++kk) {
            // Adjust index of particle neighbor
            int const k = neigh_list[kk];
            int const kSpecies = species[k];

            // cutoff between ik and jk
            double const rcutik = rcut_2D_[iSpecies][kSpecies];
            double const rcutjk = rcut_2D_[jSpecies][kSpecies];

            // Compute rik, rjk and their squares
            double rik[DIM];
            double rjk[DIM];

            for (int dim = 0; dim < DIM; ++dim) {
                rik[dim] = coordinates[k][dim] - coordinates[index][dim];
                rjk[dim] = coordinates[k][dim] - coordinates[j][dim];
            }

            double const riksq = rik[0] * rik[0] + rik[1] * rik[1] + rik[2] * rik[2];
            double const rjksq = rjk[0] * rjk[0] + rjk[1] * rjk[1] + rjk[2] * rjk[2];
            double const rikmag = std::sqrt(riksq);
            double const rjkmag = std::sqrt(rjksq);

            // Check whether three-dody not interacting
            if (rikmag > rcutik) { continue; }

            double const rvec[3] = {rijmag, rikmag, rjkmag};
            double const rcutvec[3] = {rcutij, rcutik, rcutjk};

            // Loop over descriptors
            // three-body descriptors
            for (size_t p = 0; p < name_.size(); ++p) {
                if (name_[p] != 4 && name_[p] != 5) { continue; }
                int idx = starting_index_[p];

                // Loop over same descriptor but different parameter set
                for (int q = 0; q < num_param_sets_[p]; ++q) {
                    double gc = 0.0;

                    if (name_[p] == 4) {
                        double zeta = params_[p](q, 0);
                        double lambda = params_[p](q, 1);
                        double eta = params_[p](q, 2);
                        sym_g4(zeta, lambda, eta, rvec, rcutvec, gc);
                    } else if (name_[p] == 5) {
                        double zeta = params_[p](q, 0);
                        double lambda = params_[p](q, 1);
                        double eta = params_[p](q, 2);

                        sym_g5(zeta, lambda, eta, rvec, rcutvec, gc);
                    }

                    desc[idx] += gc;
                    ++idx;
                }
            }
        }
    }
}


SymmetryFunctions::SymmetryFunctions(std::string &file_name) {
    initFromFile(file_name);
}

void SymmetryFunctions::initFromFile(std::string &file_name) {

    // Open the descriptor file
    std::ifstream file = FileIOUtils::open_file(file_name);

    // String containing data line and list of parameters
    std::vector<std::string> string_params;
    std::vector<double> double_params;
    std::vector<int> int_params;
    std::vector<bool> bool_params;
    std::string line;

    // Params
    std::string cutoff_type;
    int num_species;
    std::vector<std::string> species;
    std::map<std::pair<std::string, std::string>, double> species_pair_cutoffs;
    bool normalize = false;
    int descriptor_size;
    std::vector<std::string> symmetry_function_types;
    std::map<std::string, std::vector<double>> symmetry_function_params;

    // read Cutoff type name
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_string_params(line, string_params, 1);
    cutoff_type = string_params[0];
    string_params.clear();
    line.clear();

    // read number of species
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_int_params(line, int_params, 1);
    num_species = int_params[0];
    int_params.clear();
    line.clear();

    // read species names and cutoffs
    std::set<std::string> species_set;
    std::pair<std::string, std::string> species_species_pair;
    int species_index = num_species;
    do {
        FileIOUtils::get_next_data_line(file, line);
        FileIOUtils::parse_string_params(line, string_params, 2);
        FileIOUtils::parse_double_params(line, double_params, 1);

        species_set.insert(string_params[0]);
        species_set.insert(string_params[1]);

        species_species_pair = std::make_pair(string_params[0], string_params[1]);
        species_pair_cutoffs[species_species_pair] = double_params[0];

        species_species_pair = std::make_pair(string_params[1], string_params[0]);
        species_pair_cutoffs[species_species_pair] = double_params[0];

        double_params.clear();
        string_params.clear();
        line.clear();
    } while (--species_index > 0);
    species = std::vector<std::string>(species_set.begin(), species_set.end());

    // number of symmetry function types
    int num_symmetry_function_types;
    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_int_params(line, int_params, 1);
    num_symmetry_function_types = int_params[0];
    int_params.clear();
    line.clear();

    // Read symmetry function types and their parameters
    std::map<std::string, std::pair<int, int>> symmetry_function_type_param_sizes;

    for (int i = 0; i < num_symmetry_function_types; i++) {
        FileIOUtils::get_next_data_line(file, line);
        FileIOUtils::parse_string_params(line, string_params, 1);
        symmetry_function_types.push_back(string_params[0]);
        string_params.clear();

        FileIOUtils::parse_int_params(line, int_params, 2);
        symmetry_function_type_param_sizes[symmetry_function_types[i]] = std::make_pair(int_params[0], int_params[1]);
        int_params.clear();
        line.clear();

        symmetry_function_params[symmetry_function_types[i]].reserve(
                    symmetry_function_type_param_sizes[symmetry_function_types[i]].first *
                    symmetry_function_type_param_sizes[symmetry_function_types[i]].second);

        for (int j = 0; j < symmetry_function_type_param_sizes[symmetry_function_types[i]].first; j++) {
            FileIOUtils::get_next_data_line(file, line);
            FileIOUtils::parse_double_params(line, double_params,
                                             symmetry_function_type_param_sizes[symmetry_function_types[i]].second);
            for (int k = 0; k < symmetry_function_type_param_sizes[symmetry_function_types[i]].second; k++) {
                symmetry_function_params[symmetry_function_types[i]].push_back(double_params[k]);
            }

            double_params.clear();
            line.clear();
        }
    }

    int_params.clear();
    double_params.clear();
    string_params.clear();
    line.clear();

    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_bool_params(line, bool_params, 1);
    normalize = bool_params[0];
    bool_params.clear();
    line.clear();

    FileIOUtils::get_next_data_line(file, line);
    FileIOUtils::parse_int_params(line, int_params, 1);
    descriptor_size = int_params[0];
    int_params.clear();
    line.clear();

    file.close();

    // Initialize the descriptor
    auto cutoff_matrix = std::make_unique<double[]>(num_species * num_species);
    for (int i = 0; i < num_species; i++) {
        for (int j = 0; j < num_species; j++) {
            species_species_pair = std::make_pair(species[i], species[j]);
            cutoff_matrix[i * num_species + j] = species_pair_cutoffs[species_species_pair];
        }
    }

    // set cut offs
    n_species = num_species;
    set_species(species);

    set_cutoff(cutoff_type.c_str(), num_species, cutoff_matrix.get());

    // set symmetry functions and their parameters and width
    int _width = 0;
    for (int i = 0; i < num_symmetry_function_types; i++) {
        add_descriptor(symmetry_function_types[i].c_str(),
                       symmetry_function_params[symmetry_function_types[i]].data(),
                       symmetry_function_type_param_sizes[symmetry_function_types[i]].first,
                       symmetry_function_type_param_sizes[symmetry_function_types[i]].second);
        _width += symmetry_function_type_param_sizes[symmetry_function_types[i]].first;

    }
    width = _width;
}


void SymmetryFunctions::clone_empty(DescriptorKind *descriptorKind) {
    auto d_sf = dynamic_cast<SymmetryFunctions *>(descriptorKind);
    name_ = d_sf->name_;
    params_ = d_sf->params_;
    rcut_2D_ = d_sf->rcut_2D_;
    has_three_body_ = d_sf->has_three_body_;
    width = d_sf->width;
    num_param_sets_ = d_sf->num_param_sets_;
    num_params_ = d_sf->num_params_;
    // set params to zero, to differentiate against
    for (int i = 0; i < name_.size(); i++) {
        for (int j = 0; j < num_param_sets_[i]; j++) {
            for (int k = 0; k < num_params_[i]; k++) {
                params_[i](j, k) = 0.0;
            }
        }
    }
}

SymmetryFunctions::SymmetryFunctions(std::vector<std::string> *species, std::string *cutoff_function,
                                     double *cutoff_matrix, std::vector<std::string> *symmetry_function_types,
                                     std::vector<int> *symmetry_function_sizes,
                                     std::vector<double> *symmetry_function_parameters) {
    // set cut offs
    n_species = species->size();
    set_species(*species);

    set_cutoff(cutoff_function->c_str(), n_species, cutoff_matrix);

    // set symmetry functions and their parameters
    int offset = 0;
    int _width = 0;
    for (int i = 0; i < symmetry_function_types->size(); i++) {
        add_descriptor(symmetry_function_types->at(i).c_str(),
                       symmetry_function_parameters->data() + offset,
                       symmetry_function_sizes->at(2 * i), symmetry_function_sizes->at(2 * i + 1));
        offset += symmetry_function_sizes->at(2 * i) * symmetry_function_sizes->at(2 * i + 1);
        _width += symmetry_function_sizes->at(2 * i);
    }
    width = _width;
}

// Xi.cpp

//Xi::Xi(int q, double cutoff, std::vector<std::string> &species, std::string &radial_basis) {
//    this->cutoff = cutoff;
//    this->radial_basis = radial_basis;
//    this->q = q;
//    this->species_ = species;
//    this->width = get_width();
//    this->ln_params.resize(this->width * 6);
//    copy_params(this->q, this->ln_params.data()); // get the parameters from the file }
//    this->l_max_sq = (q + 1) * (q + 1);
//    if (radial_basis != "bessel") {
//        throw std::runtime_error("Radial basis " + radial_basis + " not implemented");
//    }
//    allocate_memory();
    //    this->create_clebsh_gordon_array();
//}
//
//void Xi::create_clebsh_gordon_array() {
//}
//
//int Xi::get_width() {
//    return get_param_length(q);
//}
//
//void Xi::compute(int index,
//                 int n_atoms,
//                 int *species,
//                 int *neighbor_lists,
//                 int number_of_neighbors,
//                 double *coordinates,
//                 double *desc) {
//    auto *coordinates_ = (VectorOfSize3 *) coordinates;
//    std::array<double, 3> r_i = {coordinates_[index][0], coordinates_[index][1], coordinates_[index][2]};
//    int ind_factor1 = (2 * q + 1);
//    int ind_factor2 = (q + 1) * (2 * q + 1);
//
//    //1. get center shifted coordinates for index atom. Neighbors exclude the center atom
//    auto i_coordinates = (VectorOfSize3 *) center_shifted_neighbor_coordinates.data();
//    bool rigid_shift = false;
//    for (int i = 0; i < number_of_neighbors; i++) {
//        int neighbor_index = neighbor_lists[i];
//        std::array<double, 3> r_j = {coordinates_[neighbor_index][0], coordinates_[neighbor_index][1],
//                                     coordinates_[neighbor_index][2]};
//        i_coordinates[i][0] = r_j[0] - r_i[0];
//        i_coordinates[i][1] = r_j[1] - r_i[1];
//        i_coordinates[i][2] = r_j[2] - r_i[2];
//        // check if any atoms are at the singularity of the derivatives
//        // occurs if x = y = 0 -> see discussion in Appendix B of J. Stimac's Dissertation
//        // if so perturb all positions slightly
//        if (i_coordinates[i][0] == 0.0 && i_coordinates[i][1] == 0.0) {
//            rigid_shift = true;
//        }
//    }
//    // Apply the rigid shift if necessary
//    if (rigid_shift) {
//        for (int i = 0; i < number_of_neighbors; i++) {
//            i_coordinates[i][0] += 1e-6;
//            i_coordinates[i][1] += 1e-6;
//            i_coordinates[i][2] += 1e-6;
//        }
//    }
//
//    // 2. get spherical harmonics for center shifted coordinates
//    // 3. map coordinates to spherical coordinates
//
//    for (int i = 0; i < number_of_neighbors; i++) {
//        r_ij[i] = std::sqrt(i_coordinates[i][0] * i_coordinates[i][0] + i_coordinates[i][1] * i_coordinates[i][1] +
//                            i_coordinates[i][2] * i_coordinates[i][2]);
//        std::array<double, 3> i_coords = {i_coordinates[i][0], i_coordinates[i][1], i_coordinates[i][2]};
//        Ylmi_all_l_from_r(q, i_coords.data(),
//                          Ylmi_real.data() + i * l_max_sq,
//                          Ylmi_imag.data() + i * l_max_sq);
//    }
//
//    // 4. get the radial basis function values
//    bessel_basis(q, cutoff, number_of_neighbors,
//                 r_ij.data(), static_cast<int>(gnl.size()), gnl.data());
//
//    // 5. compute ct and st
//    for (int n = 0; n < q + 1; n++) {
//        for (int l = 0; l < q + 1; l++) {
//            for (int m = 0; m < 2 * l + 1; m++) {
//                int idx = n * ind_factor2 + l * ind_factor1 + m;
//                ct[idx] = 0.0;
//                st[idx] = 0.0;
//                for (int i = 0; i < number_of_neighbors; i++) {
//                    ct[idx] +=
//                            gnl[i * l_max_sq + n * (q + 1) + l] *
//                            Ylmi_real[i * l_max_sq + l * l + m];
//                    st[idx] +=
//                            gnl[i * l_max_sq + n * (q + 1) + l] *
//                            Ylmi_imag[i * l_max_sq + l * l + m];
//                }
//            }
//        }
//    }
//
//    // 6. compute the descriptor
//    for (int d = 0; d < width; d++) {
//        int n0 = ln_params[d * 6 + 0];
//        int n1 = ln_params[d * 6 + 1];
//        int n2 = ln_params[d * 6 + 2];
//        int l0 = ln_params[d * 6 + 3];
//        int l1 = ln_params[d * 6 + 4];
//        int l2 = ln_params[d * 6 + 5];
//        double xi_desc = 0.0;
//        for (int m0 = -l0; m0 < l0 + 1; m0++) {
//            int m0_ind = m0 + l0;
//            int m1l = (m0 - l1 - l2 + std::abs(l1 - l2 + m0)) / 2;
//            int m1n = (m0 + l1 + l2 - std::abs(l1 - l2 - m0)) / 2;
//            for (int m1 = m1l; m1 < m1n + 1; m1++) {
//                int m2 = m0 - m1;
//                int m1_ind = m1 + l1;
//                int m2_ind = m2 + l2;
//                double C = clebsh_gordon(l1, l2, l0, m1, m2, m0);
//                xi_desc += C * (ct[n0 * ind_factor2 + l0 * ind_factor1 + m0_ind] *
//                                (ct[n1 * ind_factor2 + l1 * ind_factor1 + m1_ind] *
//                                 ct[n2 * ind_factor2 + l2 * ind_factor1 + m2_ind] -
//                                 st[n1 * ind_factor2 + l1 * ind_factor1 + m1_ind] *
//                                 st[n2 * ind_factor2 + l2 * ind_factor1 + m2_ind]) +
//                                st[n0 * ind_factor2 + l0 * ind_factor1 + m0_ind] *
//                                (ct[n1 * ind_factor2 + l1 * ind_factor1 + m1_ind] *
//                                 st[n2 * ind_factor2 + l2 * ind_factor1 + m2_ind] +
//                                 st[n1 * ind_factor2 + l1 * ind_factor1 + m1_ind] *
//                                 ct[n2 * ind_factor2 + l2 * ind_factor1 + m2_ind]));
//            }
//        }
//        double coeff = std::pow(-1.0, l0) / std::sqrt(2 * l0 + 1);
//        xi_desc *= coeff;
//        desc[d] = xi_desc;
//    }
//}
//
//void Xi::clone_empty(DescriptorKind *descriptorKind) {
//    auto *dxi = dynamic_cast<Xi *>(descriptorKind);
//    q = dxi->q;
//    cutoff = dxi->cutoff;
//    species_ = dxi->species_;
//    radial_basis = dxi->radial_basis;
//    width = dxi->width;
//    ln_params = dxi->ln_params;
//    l_max_sq = dxi->l_max_sq;
//    ln_params = dxi->ln_params;
//    allocate_memory();
//    // std::vector.resize sometimes does not work with Enzyme, so use assign instead
//}
//
//void Xi::allocate_memory() {
//    Ylmi_real.resize(MAX_NEIGHBORS * (q + 1) * (q + 1));
//    Ylmi_imag.resize(MAX_NEIGHBORS * (q + 1) * (q + 1));
//    center_shifted_neighbor_coordinates.resize(3 * MAX_NEIGHBORS);
//    center_shifted_neighbor_coordinates_zj.resize(3 * MAX_NEIGHBORS);
//    i_coordinates_spherical.resize(3 * MAX_NEIGHBORS);
//    r_ij.resize(MAX_NEIGHBORS);
//    gnl.resize(MAX_NEIGHBORS * l_max_sq);
//    ct.resize(l_max_sq * (2 * q + 1));
//    st.resize(l_max_sq * (2 * q + 1));
//
//}
//
//Xi::Xi(std::string &filename) {
//
//    // Open the descriptor file
//    std::ifstream file = FileIOUtils::open_file(filename);
//
//    // String containing data line and list of parameters
//    std::vector<std::string> string_params;
//    std::vector<double> double_params;
//    std::vector<int> int_params;
//    std::vector<bool> bool_params;
//    std::string line;
//
//    // params
//    int q_, n_species_;
//    double cutoff_;
//    std::vector<std::string> species;
//    std::string radial_basis_;
//
//    // File format:
//    /*
//     * # q
//     * 2
//     * # cutoff
//     * 5.0
//     * # radial_basis
//     * bessel
//     * # species
//     * 4
//     * H C N O
//     */
//
//    // q
//    FileIOUtils::get_next_data_line(file, line);
//    FileIOUtils::parse_int_params(line, int_params, 1);
//    q_ = int_params[0];
//    int_params.clear();
//    line.clear();
//
//    // cutoff
//    FileIOUtils::get_next_data_line(file, line);
//    FileIOUtils::parse_double_params(line, double_params, 1);
//    cutoff_ = double_params[0];
//    double_params.clear();
//    line.clear();
//
//    // radial_basis
//    FileIOUtils::get_next_data_line(file, line);
//    FileIOUtils::parse_string_params(line, string_params, 1);
//    radial_basis_ = string_params[0];
//    string_params.clear();
//    line.clear();
//
//    // species
//    FileIOUtils::get_next_data_line(file, line);
//    FileIOUtils::parse_int_params(line, int_params, 1);
//    n_species_ = int_params[0];
//    int_params.clear();
//    line.clear();
//
//    FileIOUtils::parse_string_params(line, string_params, n_species_);
//    for (auto &species_name: string_params) {
//        species.push_back(species_name);
//    }
//    string_params.clear();
//    line.clear();
//
//    file.close();
//
//    // Initialize the descriptor
//    this->cutoff = cutoff_;
//    this->radial_basis = radial_basis_;
//    this->q = q_;
//    this->species_ = species;
//    this->width = get_width();
//    this->ln_params.resize(this->width * 6);
//    copy_params(this->q, this->ln_params.data()); // get the parameters from the file }
//    this->l_max_sq = (q + 1) * (q + 1);
//    if (radial_basis != "bessel") {
//        throw std::runtime_error("Radial basis " + radial_basis + " not implemented");
//    }
//
//    allocate_memory();
//}

// math functions
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

inline double chebev(const double *c, const int m, const double x) {
    double d = 0.0, dd = 0.0, sv;
    int j;
    for (j = m - 1; j > 0; j--) {
        sv = d;
        d = 2. * x * d - dd + c[j];
        dd = sv;
    }
    return x * d - dd + 0.5 * c[0];
}

// From numerical recipies in C++ 3rd edition
// omiting the besselk and derivative parts
double bessel_I(const double nu, const double x) {
    const int MAXIT = 10000;
    const double EPS = std::numeric_limits<double>::epsilon();
    const double FPMIN = std::numeric_limits<double>::min() / EPS;
    const double XMIN = 2.0;
    double a, a1, b, c, d, del, del1, delh, dels, e, f, fact, fact2, ff, gam1, gam2, gammi, gampl, h, p, pimu, q, q1,
            q2, qnew, ril, ril1, rimu, rip1, ripl, ritemp, rk1, rkmu, rkmup, rktemp, s, sum, sum1, x2, xi, xi2, xmu, xmu2, xx;
    int i, l, nl;

    static double c1[] = {
            -1.142022680371168e0, 6.5165112670737e-3, 3.087090173086e-4, -3.4706269649e-6,
            6.9437664e-9, 3.67795e-11, -1.356e-13};
    static double c2[] = {
            1.843740587300905e0, -7.68528408447867e-2, 1.2719271366546e-3, -4.9717367042e-6,
            -3.31261198e-8, 2.423096e-10, -1.702e-13, -1.49e-15};

    if (x <= 0.0 || nu < 0.0) {
        throw std::runtime_error("bad arguments in besselik");
    }
    // nl = static_cast<int>(nu + 0.5);
    nl = static_cast<int>(std::lround(nu + 0.5));
    xmu = nu - nl;
    xmu2 = xmu * xmu;
    xi = 1.0 / x;
    xi2 = 2.0 * xi;
    h = nu * xi;
    if (h < FPMIN) h = FPMIN;
    b = xi2 * nu;
    d = 0.0;
    c = h;
    for (i = 0; i < MAXIT; i++) {
        b += xi2;
        d = 1.0 / (b + d);
        c = b + 1.0 / c;
        del = c * d;
        h = del * h;
        if (std::abs(del - 1.0) <= EPS) break;
    }
    if (i >= MAXIT) {
        throw std::runtime_error("x too large in besselik; try asymptotic expansion");
    }
    ril = FPMIN;
    ripl = h * ril;
    ril1 = ril;
    rip1 = ripl;
    fact = nu * xi;
    for (l = nl - 1; l >= 0; l--) {
        ritemp = fact * ril + ripl;
        fact -= xi;
        ripl = fact * ritemp + ril;
        ril = ritemp;
    }
    f = ripl / ril;
    if (x < XMIN) {
        x2 = 0.5 * x;
        pimu = M_PI * xmu;
        fact = (std::abs(pimu) < EPS ? 1.0 : pimu / std::sin(pimu));
        d = -std::log(x2);
        e = xmu * d;
        fact2 = (std::abs(e) < EPS ? 1.0 : std::sinh(e) / e);
        xx = 8.0 * (xmu * xmu) - 1.0;
        gam1 = chebev(c1, 7, xx); // double precision run summation for 7 and 8
        gam2 = chebev(c2, 8, xx);
        gampl = gam2 - xmu * gam1;
        gammi = gam2 + xmu * gam1;
        ff = fact * (gam1 * std::cosh(e) + gam2 * fact2 * d);
        sum = ff;
        e = std::exp(e);
        p = 0.5 * e / gampl;
        q = 0.5 / (e * gammi);
        c = 1.0;
        d = x2 * x2;
        sum1 = p;
        for (i = 1; i <= MAXIT; i++) {
            ff = (i * ff + p + q) / (i * i - xmu2);
            c *= (d / i);
            p /= (i - xmu);
            q /= (i + xmu);
            del = c * ff;
            sum += del;
            del1 = c * (p - i * ff);
            sum1 += del1;
            if (std::abs(del) < std::abs(sum) * EPS) {
                break;
            }
        }
        if (i > MAXIT) throw std::runtime_error("bessk series failed to converge");
        rkmu = sum;
        rk1 = sum1 * xi2;
    } else {
        b = 2.0 * (1.0 + x);
        d = 1.0 / b;
        h = delh = d;
        q1 = 0.0;
        q2 = 1.0;
        a1 = 0.25 - xmu2;
        q = c = a1;
        a = -a1;
        s = 1.0 + q * delh;
        for (i = 1; i < MAXIT; i++) {
            a -= 2 * i;
            c = -a * c / (i + 1.0);
            qnew = (q1 - b * q2) / a;
            q1 = q2;
            q2 = qnew;
            q += c * qnew;
            b += 2.0;
            d = 1.0 / (b + a * d);
            delh = (b * d - 1.0) * delh;
            h += delh;
            dels = q * delh;
            s += dels;
            if (std::abs(dels / s) <= EPS) {
                break;
            }
        }
        if (i >= MAXIT) {
            throw std::runtime_error("besselik: failure to converge in cf2");
        }
        h = a1 * h;
        rkmu = std::sqrt(M_PI / (2.0 * x)) * std::exp(-x) / s;
        rk1 = rkmu * (xmu + x + 0.5 - h) * xi;
    }
    rkmup = xmu * xi * rkmu - rk1;
    rimu = xi / (f * rkmu - rkmup);
    return (rimu * ril1) / ril; //po
    // double ipo=(rimu*rip1)/ril;
    // for (i=1;i <= nl;i++) {
    // rktemp=(xmu+i)*xi2*rk1+rkmu;
    // rkmu=rk1;
    // rk1=rktemp;
    // }
    // double ko=rkmu;
    // double kpo=nu*xi*rkmu-rk1;
    // double xik = x;
    // double nuik = nu;
}


double bessel_J(const double nu, const double x) {
    const int MAXIT = 10000;
    const double EPS = std::numeric_limits<double>::epsilon();
    const double FPMIN = std::numeric_limits<double>::min() / EPS;
    const double XMIN = 2.0;
    double a, b, br, bi, c, cr, ci, d, del, del1, den, di, dlr, dli, dr, e, f, fact, fact2, fact3, ff, gam, gam1, gam2,
    gammi, gampl, h, p, pimu, pimu2, q, r, rjl, rjl1, rjmu, rjp1, rjpl, rjtemp, ry1, rymu, rymup, rytemp, sum, sum1,
    temp, w, x2, xi, xi2, xmu, xmu2, xx;
    int i, isign, l, nl;
    const double c1[7] = {-1.142022680371168e0, 6.5165112670737e-3,
                          3.087090173086e-4, -3.4706269649e-6, 6.9437664e-9, 3.67795e-11,
                          -1.356e-13};
    const double c2[8] = {1.843740587300905e0, -7.68528408447867e-2,
                          1.2719271366546e-3, -4.9717367042e-6, -3.31261198e-8, 2.423096e-10,
                          -1.702e-13, -1.49e-15};

    if (x <= 0.0 || nu < 0.0) throw std::runtime_error("bad arguments in besseljy");
    nl = (x < XMIN ? static_cast<int>(std::lround(nu + 0.5)) : std::max(0, int(nu - x + 1.5)));
    xmu = nu - nl;
    xmu2 = xmu * xmu;
    xi = 1.0 / x;
    xi2 = 2.0 * xi;
    w = xi2 / M_PI;
    isign = 1;
    h = nu * xi;
    if (h < FPMIN) h = FPMIN;
    b = xi2 * nu;
    d = 0.0;
    c = h;
    for (i = 0; i < MAXIT; i++) {
        b += xi2;
        d = b - d;
        if (std::abs(d) < FPMIN) d = FPMIN;
        c = b - 1.0 / c;
        if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        del = c * d;
        h = del * h;
        if (d < 0.0) isign = -isign;
        if (std::abs(del - 1.0) <= EPS) break;
    }
    if (i >= MAXIT)
        throw std::runtime_error("x too large in besseljy; try asymptotic expansion");
    rjl = isign * FPMIN;
    rjpl = h * rjl;
    rjl1 = rjl;
    rjp1 = rjpl;
    fact = nu * xi;
    for (l = nl - 1; l >= 0; l--) {
        rjtemp = fact * rjl + rjpl;
        fact -= xi;
        rjpl = fact * rjtemp - rjl;
        rjl = rjtemp;
    }
    if (rjl == 0.0) rjl = EPS;
    f = rjpl / rjl;
    if (x < XMIN) {
        x2 = 0.5 * x;
        pimu = M_PI * xmu;
        fact = (std::abs(pimu) < EPS ? 1.0 : pimu / std::sin(pimu));
        d = -std::log(x2);
        e = xmu * d;
        fact2 = (std::abs(e) < EPS ? 1.0 : sinh(e) / e);
        xx = 8.0 * (xmu * xmu) - 1.0;
        gam1 = chebev(c1, 7, xx);
        gam2 = chebev(c2, 8, xx);
        gampl = gam2 - xmu * gam1;
        gammi = gam2 + xmu * gam1;
        ff = 2.0 / M_PI * fact * (gam1 * std::cosh(e) + gam2 * fact2 * d);
        e = std::exp(e);
        p = e / (gampl * M_PI);
        q = 1.0 / (e * M_PI * gammi);
        pimu2 = 0.5 * pimu;
        fact3 = (std::abs(pimu2) < EPS ? 1.0 : std::sin(pimu2) / pimu2);
        r = M_PI * pimu2 * fact3 * fact3;
        c = 1.0;
        d = -x2 * x2;
        sum = ff + r * q;
        sum1 = p;
        for (i = 1; i <= MAXIT; i++) {
            ff = (i * ff + p + q) / (i * i - xmu2);
            c *= (d / i);
            p /= (i - xmu);
            q /= (i + xmu);
            del = c * (ff + r * q);
            sum += del;
            del1 = c * p - i * del;
            sum1 += del1;
            if (std::abs(del) < (1.0 + std::abs(sum)) * EPS) break;
        }
        if (i > MAXIT) throw std::runtime_error("bessy series failed to converge");
        rymu = -sum;
        ry1 = -sum1 * xi2;
        rymup = xmu * xi * rymu - ry1;
        rjmu = w / (rymup - f * rymu);
    } else {
        a = 0.25 - xmu2;
        p = -0.5 * xi;
        q = 1.0;
        br = 2.0 * x;
        bi = 2.0;
        fact = a * xi / (p * p + q * q);
        cr = br + q * fact;
        ci = bi + p * fact;
        den = br * br + bi * bi;
        dr = br / den;
        di = -bi / den;
        dlr = cr * dr - ci * di;
        dli = cr * di + ci * dr;
        temp = p * dlr - q * dli;
        q = p * dli + q * dlr;
        p = temp;
        for (i = 1; i < MAXIT; i++) {
            a += 2 * i;
            bi += 2.0;
            dr = a * dr + br;
            di = a * di + bi;
            if (std::abs(dr) + std::abs(di) < FPMIN) dr = FPMIN;
            fact = a / (cr * cr + ci * ci);
            cr = br + cr * fact;
            ci = bi - ci * fact;
            if (std::abs(cr) + std::abs(ci) < FPMIN) cr = FPMIN;
            den = dr * dr + di * di;
            dr /= den;
            di /= -den;
            dlr = cr * dr - ci * di;
            dli = cr * di + ci * dr;
            temp = p * dlr - q * dli;
            q = p * dli + q * dlr;
            p = temp;
            if (std::abs(dlr - 1.0) + std::abs(dli) <= EPS) break;
        }
        if (i >= MAXIT) throw std::runtime_error("cf2 failed in besseljy");
        gam = (p - f) / q;
        rjmu = std::sqrt(w / ((p - f) * gam + q));
        rjmu = SIGN(rjmu, rjl);
        rymu = rjmu * gam;
        rymup = rymu * (p + q / gam);
        // ry1 = xmu * xi * rymu - rymup;
    }
    fact = rjmu / rjl;
    return rjl1 * fact;
    // jo=rjl1*fact;
    // jpo=rjp1*fact;
    // for (i=1;i<=nl;i++) {
    // rytemp=(xmu+i)*xi2*ry1-rymu;
    // rymu=ry1;
    // ry1=rytemp;
}
// yo=rymu;
// ypo=nu*xi*rymu-ry1;
// xjy = x;
// nujy = nu;

double spherical_in(double n, double x) {
    //Modified spherical Bessel function of the first kind
    return bessel_I(n + 0.5, x) * sqrt(M_PI / (2 * x));
}
double spherical_in(int n, double x){
    return spherical_in(static_cast<double>(n), x);
}

double spherical_jn(double n, double x) {
    //Spherical Bessel function of the first kind
    return bessel_J(n + 0.5, x) * sqrt(M_PI / (2 * x));
}
double spherical_jn(int n, double x){
    return spherical_jn(static_cast<double>(n), x);
}

double halleys_root(double l, double lwr_bnd, double upr_bnd){
    double TOL1=2.2204e-13, TOL2=2.2204e-14;
    int MAXIT=100000;

    double x=(lwr_bnd + upr_bnd)/2.;
    double l1 = l+1;

    int i = 0;
    throw std::runtime_error("halleys_root() not implemented.");
//    while (i < MAXIT){
//        double a = spherical_jn(l, x);
//        double b = spherical_jn(l1, x);
//
//        if (std::abs(a) < TOL2) {
//            break;
//        }
//
//        double x2 = x*x;
//        double dx = -2*x*a*(l*a-x*b) / ((l*l1+x2)*a*a - 2*x*(2*l+1)*a*b + 2*x2*b*b);
//        if (std::abs(dx) < TOL1) {
//            break;
//        }
//
//        if (dx > 0) {
//            lwr_bnd = x;
//        } else {
//            upr_bnd = x;
//        }
//
//        if ((upr_bnd-x) < dx || dx < (lwr_bnd-x)) {
//            dx = (upr_bnd - lwr_bnd) / 2. - x;
//        }
//
//        x=x+dx;
//        i++;
//    }

    if (i>MAXIT-1) {
        throw std::runtime_error("halleys_root() failed to converge.");
    }
    return x;
}

double halleys_root(int l, double lwr_bnd, double upr_bnd){
    return halleys_root(static_cast<double>(l), lwr_bnd, upr_bnd);
}


// spherical jn zeros precomputed, column major j
void spherical_jn_zeros(int n_max, double * u_all ){
    // Expected u_all to be a (n_max+2, n_max+1)
    if (n_max > precomputed_values::bessel_zeros_max_j) {
        throw std::runtime_error("n_max is too large for precomputed values.");
    }
    for (int n = 0; n < n_max + 2; n++) {
        for (int j = 0 ; j < n_max + 1; j++) {
            u_all[n * (n_max + 1) + j] = precomputed_values::bessel_zeros[j][n];
        }
    }
}

// gl_quad.cpp

std::vector<double> get_gl_weights() {
    return {7.34634490505672E-4, 0.001709392653518105, 0.002683925371553482, 0.003655961201326375,
            0.00462445006342212, 0.00558842800386552, 0.00654694845084532, 0.007499073255464712,
            0.008443871469668971, 0.00938041965369446, 0.01030780257486897, 0.01122511402318598,
            0.012131457662979497, 0.0130259478929715423, 0.0139077107037187727, 0.014775884527441302,
            0.015629621077546003, 0.01646808617614521, 0.017290460568323582, 0.018095940722128117,
            0.018883739613374905, 0.019653087494435306, 0.020403232646209433, 0.021133442112527642,
            0.02184300241624739, 0.022531220256336273, 0.023197423185254122, 0.023840960265968206,
            0.02446120270795705, 0.02505754448157959, 0.02562940291020812, 0.02617621923954568,
            0.026697459183570963, 0.02719261344657688, 0.027661198220792388, 0.028102755659101173,
            0.028516854322395098, 0.028903089601125203, 0.029261084110638277, 0.02959048805991264,
            0.029890979593332831, 0.03016226510516914, 0.03040407952645482, 0.03061618658398045,
            0.03079837903115259, 0.030950478850490988, 0.031072337427566517, 0.031163835696209907,
            0.03122488425484936, 0.03125542345386336, 0.031255423453863357, 0.03122488425484936,
            0.0311638356962099068, 0.031072337427566517, 0.030950478850490988, 0.03079837903115259,
            0.030616186583980449, 0.03040407952645482, 0.0301622651051691449, 0.02989097959333283,
            0.029590488059912643, 0.029261084110638277, 0.028903089601125203, 0.0285168543223951,
            0.02810275565910117, 0.02766119822079239, 0.02719261344657688, 0.02669745918357096,
            0.02617621923954568, 0.025629402910208116, 0.02505754448157959, 0.024461202707957053,
            0.02384096026596821, 0.023197423185254122, 0.0225312202563362727, 0.021843002416247386,
            0.02113344211252764, 0.020403232646209433, 0.019653087494435306, 0.0188837396133749046,
            0.018095940722128117, 0.017290460568323582, 0.016468086176145213, 0.015629621077546003,
            0.0147758845274413, 0.013907710703718773, 0.01302594789297154, 0.0121314576629795,
            0.01122511402318598, 0.01030780257486897, 0.00938041965369446, 0.008443871469668971,
            0.007499073255464712, 0.00654694845084532, 0.005588428003865515, 0.00462445006342212,
            0.003655961201326375, 0.002683925371553482, 0.001709392653518105, 7.3463449050567E-4};
}

std::vector<double> get_orig_gl_grid(){
    return {-0.999713726773441234, -0.998491950639595818, -0.996295134733125149, -0.99312493703744346,
            -0.98898439524299175, -0.98387754070605702, -0.97780935848691829, -0.97078577576370633,
            -0.962813654255815527, -0.95390078292549174, -0.94405587013625598, -0.933288535043079546,
            -0.921609298145333953, -0.90902957098252969, -0.895561644970726987, -0.881218679385018416,
            -0.86601468849716462, -0.849964527879591284, -0.833083879888400824, -0.815389238339176254,
            -0.79689789239031448, -0.77762790964949548, -0.757598118519707176, -0.736828089802020706,
            -0.715338117573056447, -0.69314919935580197, -0.670283015603141016, -0.64676190851412928,
            -0.622608860203707772, -0.59784747024717872, -0.57250193262138119, -0.546597012065094168,
            -0.520158019881763057, -0.493210789208190934, -0.465781649773358042, -0.437897402172031513,
            -0.409585291678301543, -0.380872981624629957, -0.351788526372421721, -0.322360343900529152,
            -0.292617188038471965, -0.26258812037150348, -0.23230248184497397, -0.201789864095735997,
            -0.171080080538603275, -0.140203137236113973, -0.109189203580061115, -0.0780685828134366367,
            -0.046871682421591632, -0.015628984421543083, 0.0156289844215430829, 0.046871682421591632,
            0.078068582813436637, 0.109189203580061115, 0.140203137236113973, 0.171080080538603275,
            0.201789864095735997, 0.23230248184497397, 0.262588120371503479, 0.292617188038471965,
            0.322360343900529152, 0.351788526372421721, 0.380872981624629957, 0.409585291678301543,
            0.437897402172031513, 0.465781649773358042, 0.49321078920819093, 0.520158019881763057,
            0.546597012065094168, 0.572501932621381191, 0.59784747024717872, 0.622608860203707772,
            0.64676190851412928, 0.670283015603141016, 0.693149199355801966, 0.715338117573056447,
            0.736828089802020706, 0.75759811851970718, 0.77762790964949548, 0.79689789239031448,
            0.81538923833917625, 0.833083879888400824, 0.849964527879591284, 0.866014688497164623,
            0.881218679385018416, 0.89556164497072699, 0.90902957098252969, 0.921609298145333953,
            0.933288535043079546, 0.94405587013625598, 0.953900782925491743, 0.96281365425581553,
            0.970785775763706332, 0.977809358486918289, 0.983877540706057016, 0.98898439524299175,
            0.99312493703744346, 0.99629513473312515, 0.998491950639595818, 0.99971372677344123};
}

std::vector<double> get_gl_grid(double cutoff){
    // GL grid shifted to (0, cutoff)
    auto grid = get_orig_gl_grid();
    double scale = cutoff/2;
    for(double & grid_val: grid){
        grid_val = scale * (grid_val + 1); // (b-a)/2 * x + (b+a)/2
    }
    return grid;
}

// precomputed_bessel_zeros
namespace precomputed_values {
    double bessel_zeros[bessel_zeros_max_j + 1][bessel_zeros_max_roots] = {
{3.1415926535897932385, 6.2831853071795864769, 9.4247779607693797153, 12.566370614359172954, 15.707963267948966192, 18.849555921538759431, 21.991148575128552669, 25.132741228718345908, 28.274333882308139146, 31.415926535897932385, 34.557519189487725623, 37.699111843077518862, 40.8407044966673121, 43.982297150257105338, 47.123889803846898577, 50.265482457436691815, 53.407075111026485054, 56.548667764616278292, 59.690260418206071531, 62.831853071795864769, 65.973445725385658008, 69.115038378975451246, 72.256631032565244485, 75.398223686155037723, 78.539816339744830962, 81.6814089933346242, 84.823001646924417438, 87.964594300514210677, 91.106186954104003915, 94.247779607693797154, 97.389372261283590392, 100.53096491487338363, 103.67255756846317687, 106.81415022205297011, 109.95574287564276335, 113.09733552923255658, 116.23892818282234982, 119.38052083641214306, 122.5221134900019363, 125.66370614359172954, 128.80529879718152278, 131.94689145077131602, 135.08848410436110925, 138.23007675795090249, 141.37166941154069573, 144.51326206513048897, 147.65485471872028221, 150.79644737231007545, 153.93804002589986868, 157.07963267948966192, 160.22122533307945516, 163.3628179866692484, 166.50441064025904164, 169.64600329384883488, 172.78759594743862812, 175.92918860102842135, 179.07078125461821459, 182.21237390820800783, 185.35396656179780107, 188.49555921538759431, 191.63715186897738755, 194.77874452256718078, 197.92033717615697402, 201.06192982974676726, 204.2035224833365605, 207.34511513692635374, 210.48670779051614698, 213.62830044410594022, 216.76989309769573345, 219.91148575128552669, 223.05307840487531993, 226.19467105846511317, 229.33626371205490641, 232.47785636564469965, 235.61944901923449288, 238.76104167282428612, 241.90263432641407936, 245.0442269800038726, 248.18581963359366584, 251.32741228718345908, 254.46900494077325232, 257.61059759436304555, 260.75219024795283879, 263.89378290154263203, 267.03537555513242527, 270.17696820872221851, 273.31856086231201175, 276.46015351590180498, 279.60174616949159822, 282.74333882308139146, 285.8849314766711847, 289.02652413026097794, 292.16811678385077118, 295.30970943744056442, 298.45130209103035765, 301.59289474462015089, 304.73448739820994413, 307.87608005179973737, 311.01767270538953061, 314.15926535897932385},
{4.4934094579090641753, 7.7252518369377071637, 10.9041216594288998271, 14.06619391283147348, 17.2207552719307687396, 20.3713029592875628451, 23.5194524986890065465, 26.6660542588126735284, 29.8115987908929588368, 32.956389039822476725, 36.100622244375610697, 39.244432361164192842, 42.387913568131919856, 45.531134013991279825, 48.674144231954387139, 51.816982487279669512, 54.959678287888935871, 58.102254754495592595, 61.244730260374400373, 64.387119590557413712, 67.529434777144116904, 70.671685711619500774, 73.813880600680643258, 76.956026310331182556, 80.098128628945113291, 83.240192470723404568, 86.38222203472871439, 89.524220930417189145, 92.666192277622837621, 95.808138786861698801, 98.950062824331880296, 102.091966464907643337, 105.233851535637597908, 108.375719651674689217, 111.517572246131011892, 114.65941059502307872, 117.801235838224382593, 120.943048997151490762, 124.084850989762759958, 127.226642643334325819, 130.368424705388413154, 133.510197853078390227, 136.6519627012789845, 139.793719809585402804, 142.935469688389264414, 146.077212804170359094, 149.218949584119822713, 152.360680420191250807, 155.502405672660667024, 158.644125673263441474, 161.785840727965673579, 164.927551119418787709, 168.069257109138798825, 171.210958939445618479, 174.352656835192672042, 177.49435100531281099, 180.636041644202888235, 183.777728932966305016, 186.9194130405302413, 190.061094124652071397, 193.202772332827579759, 196.344447803111976972, 199.486120664863329871, 202.62779103941682711, 205.769459040697272812, 208.911124775776311434, 212.052788345380116152, 215.194449844352603497, 218.336109362078654144, 221.47776698287131133, 224.619422786326483955, 227.761076847648292189, 230.902729237947851826, 234.044380024517993331, 237.186029271086147027, 240.327677038047392473, 243.469323382679463835, 246.610968359341320418, 249.752612019656729566, 252.894254412684165335, 256.035895585074198362, 259.177535581215438369, 262.319174443369989046, 265.460812211799284176, 268.602448924881092552, 271.744084619218406433, 274.885719329740862866, 278.027353089799288569, 281.168985931253906217, 284.310617884556692451, 287.452248978828335042, 290.593879241930197956, 293.735508700531668085, 296.877137380173225755, 300.01876530532555248, 303.16039249944496344, 306.30201898502542852, 309.44364478364742444, 312.58526991602384086, 315.7268944020431457},
{5.7634591968945497914, 9.0950113304763551561, 12.322940970566582052, 15.5146030108867482304, 18.689036355362822202, 21.8538742227097657924, 25.0128032022896124663, 28.1678297079936238748, 31.3201417074471745358, 34.4704883312849886657, 37.6193657535884248481, 40.7671158214068047615, 43.9139818113646514499, 47.0601416127605324053, 50.205728336738035107, 53.350843585293213552, 56.495566261811978209, 59.639958579558153238, 62.784070256180135426, 65.927941502958645068, 69.071605194609605603, 72.215088470407262459, 75.358413933321402834, 78.501600560239114137, 81.644664401382347821, 84.787619123785468523, 87.930476437957069113, 91.073246436016345615, 94.215937862023574093, 97.358558329859645446, 100.501114500159072548, 103.643612225003910724, 106.786056667031694356, 109.92845239808592701, 113.07080348139526652, 116.21311354040373728, 119.355385816715570369, 122.497623219111900816, 125.639828365204358615, 128.782003616984664189, 131.924151111289095709, 135.066272786006828632, 138.208370402710185364, 141.350445566264113556, 144.492499741875148938, 147.634534269961670769, 150.776550379163540658, 153.918549197757217708, 157.060531763699817301, 160.202499033490487031, 163.344451890008460899, 166.486391149463073015, 169.62831756757094999, 172.770231845058825703, 175.912134632576345481, 179.05402653509137502, 182.195908115830318476, 185.337779899817467868, 188.479642377060197049, 191.621496005420669273, 194.76334121320947574, 197.905178401532121734, 201.047007946415409672, 204.188830200737436895, 207.33064549598204923, 210.472454143836101281, 213.614256437645714065, 216.756052653745842339, 219.897843052675827256, 223.039627880292180955, 226.181407368788599348, 229.323181737632103306, 232.464951194423245736, 235.606715935687474961, 238.748476147603998059, 241.890232006677828437, 245.031983680360118616, 248.17373132762136243, 251.315475099481592199, 254.457215139501288798, 257.598951584236359678, 260.74068456366021636, 263.8824142015556941, 267.024140615879298188, 270.165863919100030139, 273.307584218514839747, 276.449301616542562925, 279.591016210998038001, 282.73272809534794262, 285.874437358949757803, 289.016144087275143336, 292.157848362118898175, 295.299550261794579643, 298.441249861317764713, 301.582947232577854732, 304.724642444499250569, 307.866335563192657639, 311.008026652097218874, 314.149715772114117839, 317.291402981732243329},
{6.98793200050051995909, 10.4171185473793647632, 13.6980231532492489995, 16.9236212852138395789, 20.1218061744538182856, 23.304246988939651352, 26.47676366453912815, 29.6426045403158091721, 32.803732385196107943, 35.9614058047090330689, 39.1164701902715353863, 42.2695149777811615122, 45.4209639722562069707, 48.5711298516317806558, 51.7202484303878732049, 54.8685009575007805603, 58.016029064100534072, 61.1629450448140524963, 64.3093390906704893265, 67.4552844798028187062, 70.600841369236535329, 73.746059609192634974, 76.890980862079117327, 80.035640218852123887, 83.180067446675708705, 86.324287962487324627, 89.468323600292316709, 92.612193221471850136, 95.755913204366117419, 98.89949784012185194, 102.042959655106928339, 105.186309675317145483, 108.329557644603188895, 111.472712205866100025, 114.615781052354396553, 117.75877105466703576, 120.901688367896685125, 124.044538522445832213, 127.187326501347672375, 130.330056806375709615, 133.472733514794525542, 136.615360328262310629, 139.757940615123288317, 142.900477447109783712, 146.042973631297719626, 149.185431738016803081, 152.327854125300674589, 155.470242960367453389, 158.612600238543222911, 161.754927799976760444, 164.897227344440616311, 168.039500444469420935, 171.181748557049392517, 174.323973034042111042, 177.466175131499651609, 180.60835601800627148, 183.750516782163322673, 186.89265843931834675, 190.034781937625935632, 193.176888163516530111, 196.318977946639561926, 199.461052064338966045, 202.603111245711881717, 205.745156175295144894, 208.887187496418801208, 212.02920581426121288, 215.171211698636289738, 218.313205686539855681, 221.455188284479092931, 224.597159970606324241, 227.739121196676044502, 230.881072389842052382, 234.023013954309721143, 237.164946272856852481, 240.306869708235149806, 243.448784604463103663, 246.590691288019980941, 249.732590068949633329, 252.874481241881973485, 256.016365086979196271, 259.15824187081313549, 262.300111847179533757, 265.441975257854455646, 268.583832333297584508, 271.725683293306704521, 274.867528347627275977, 278.009367696520658232, 281.151201531294216797, 284.293030034796264734, 287.434853381878530389, 290.576671739828610501, 293.718485268774657187, 296.860294122064356796, 300.002098446620086153, 303.143898383271975253, 306.285694067070463523, 309.427485627579807803, 312.569273189153882835, 315.711056871195508307, 318.852836788400439152},
{8.18256145257124270171, 11.7049071545703905585, 15.0396647076165208079, 18.3012559595419902197, 21.5254177333999454374, 24.7275655478350333714, 27.9155761994213606421, 31.0939332140793071745, 34.2653900861015857833, 37.4317367682014947261, 40.5941896534211787513, 43.7536054311193607293, 46.910605490089278961, 50.0656518347345660589, 53.2190952897377382723, 56.3712071531379845072, 59.5222005873999266072, 62.6722454406628017289, 65.8214787430158364408, 68.970012285027948629, 72.1179381847262091383, 75.2653330406638503427, 78.4122610737213617657, 81.5587765341649561977, 84.7049255671969040453, 87.8507476741781883866, 90.996276868325056102, 94.141542596987295411, 97.286570483779551005, 100.431382930366768584, 103.575999607952784835, 106.720437861379711241, 109.86471304346030603, 113.008838793214684788, 116.152827268701975629, 119.296689342869047494, 122.44043476909787751, 125.584072321787319813, 128.727609916256896688, 131.871054711438312215, 135.014413198171677494, 138.157691275408282874, 141.300894316210271256, 144.444027225107070964, 147.587094488101596946, 150.730100216402685491, 153.873048184783664336, 157.015941865322336487, 160.158784457158676483, 163.301578912808245314, 166.44432796148780253, 169.587034129841712017, 172.729699760401009109, 175.872327028059420908, 179.014917954810596293, 182.15747442295700181, 185.299998186972313455, 188.442490884174816766, 191.584954044348609664, 194.727389098431701524, 197.869797386374937444, 201.012180164262651104, 204.154538610774732671, 207.296873833060115194, 210.439186872083304115, 213.581478707498307566, 216.723750262098008477, 219.866002405881516965, 223.008235959777238177, 226.15045169905518919, 229.292650356458416242, 232.434832625080130048, 235.576999161010332276, 238.719150585773199199, 241.861287488574275041, 245.003410428374569993, 248.145519935806923777, 251.287616514948456947, 254.429700644961564733, 257.571772781614691221, 260.713833358693036785, 263.855882789308383292, 266.9979214671163559, 270.139949767448665292, 273.281968048367179539, 276.423976651646051258, 279.565975903687565382, 282.707966116376868601, 285.849947587880287228, 288.99192060339153046, 292.133885435829705898, 295.275842346492739602, 298.41779158566949009, 301.559733393213571241, 304.701667999081650149, 307.843595623838759907, 310.98551647913296184, 314.127430768141504723, 317.269338685990458282, 320.411240420149642989},
{9.35581211104274617144, 12.9665301727743445575, 16.3547096393504631821, 19.6531521018211851004, 22.9045506479037219465, 26.1277501372255060346, 29.3325625785848214142, 32.5246612885788443959, 35.7075769530614145615, 38.883630955463055444, 42.0544164128268260738, 45.2210650159239052771, 48.3844038605503549851, 51.5450520425883867001, 54.7034825076868135544, 57.8600629728451069042, 61.015083772306085042, 64.1687772729670418195, 67.3213317037489544179, 70.4729011938088596429, 73.6236131825175339165, 76.7735739725347468071, 79.9228729484101693269, 83.071585821281649257, 86.2197771528090104778, 89.3675023388337649474, 92.5148091832916120496, 95.6617391580063383153, 98.8083284192693374155, 101.9546086343615367509, 105.1006076582799619507, 108.2463500914566292361, 111.391857742222656248, 114.537150012495968411, 117.682244221180031117, 120.827155876715358486, 123.971898907882496705, 127.116485860138976099, 130.260928063354829442, 133.405235775696849409, 136.54941830753025152, 139.69348412850491437, 142.837440960431915921, 145.981295858104260744, 149.125055279850201529, 152.268725149310438484, 155.412310909687800947, 158.555817571518907339, 161.699249754853235183, 164.842611726589286326, 167.985907433604772261, 171.129140532223722937, 174.272314414484740713, 177.415432231608546301, 180.558496915007289454, 183.701511195131035061, 186.844477618406929407, 189.987398562492612068, 193.130276250036490809, 196.273112761112737096, 199.415910044477625495, 202.558669927775582023, 205.701394126807567918, 208.84408425396082562, 211.986741825887234723, 215.129368270507300339, 218.271964933407900206, 221.414533083694160577, 224.557073919349054566, 227.699588572148384312, 230.842078112173604955, 233.984543551960375683, 237.126985850316697241, 240.269405915840944519, 243.411804610166965705, 246.554182750960643047, 249.696541114689849082, 252.838880439187546943, 255.981201426025840067, 259.123504742717045913, 262.265791024756324628, 265.408060877519014733, 268.550314878024594579, 271.692553576578083591, 274.834777498298706524, 277.976987144544754134, 281.119182994242773528, 284.261365505128501068, 287.40353511490630134, 290.545692242333289618, 293.687837288233785748, 296.829970636449268345, 299.972092654728564475, 303.114203695562616711, 306.256304096967812627, 309.398394183221537571, 312.540474265553316748, 315.682544642794644271, 318.82460560199035231, 321.966657418974150426},
{10.51283540809399798173, 14.2073924588424603271, 17.6479748701658975077, 20.983463068944768959, 24.2627680423970078407, 27.5078683649042514443, 30.7303807316466494193, 33.9371083026413006636, 37.1323317248601418401, 40.3188925092264005597, 43.4987571413475067163, 46.6733329249516659635, 49.8436551888161056289, 53.0105034816552853083, 56.1744764965461010826, 59.3360419634792450174, 62.4955708011774443606, 65.6533610593939573739, 68.8096550587705329221, 71.9646518901159136788, 75.1185166810816413879, 78.2713875686616276846, 81.4233810160296541958, 84.5745959163058534558, 87.7251167952320876506, 90.8750163360521675681, 94.0243573886629106201, 97.1731945821765997506, 100.3215756295174950011, 103.4695423906961086281, 106.6171317453854654094, 109.7643763136174968853, 112.9113050546266387208, 116.0579437672583545674, 119.2043155103463176197, 122.3504409576260296626, 125.4963386987946119182, 128.6420254960283599095, 131.787516503471592274, 134.932825455794139965, 138.077964830792374314, 141.222945990113715509, 144.367779301466941314, 147.51247424510208713, 150.657039506874939555, 153.801483059829402735, 156.945812235918739717, 160.090033789230087725, 163.234153951864892966, 166.378178483452456169, 169.522112715127834394, 172.665961588683490461, 175.809729691501976215, 178.953421287791083736, 182.097040346570472943, 185.240590566797490035, 188.38407539996785847, 191.527498070482622706, 194.670861594034898186, 197.814168794237593848, 200.957422317685469761, 204.100624647620962462, 207.243778116352563211, 210.386884916556676723, 213.529947111578407116, 216.672966644833267232, 219.815945348400095522, 222.958884950885244892, 226.101787084629172086, 229.244653292318727319, 232.387485033061571718, 235.53028368797310642, 238.673050565320972377, 241.815786905267480182, 244.95849388424617413, 248.101172619005054907, 251.243824170345721122, 254.386449546584789874, 257.529049706761375994, 260.671625563612109933, 263.81417798633312188, 266.956707803146585416, 270.099215803687772308, 273.241702741227098817, 276.384169334740323566, 279.526616270838870677, 282.669044205571184556, 285.811453766105061209, 288.953845552300033908, 292.096220138178108048, 295.238578073300432249, 298.38091988405685227, 301.523246074874713935, 304.665557129352755, 307.807853511325447985, 310.950135665862721713, 314.09240402020959418, 317.234658984669889542, 320.376900953437884034, 323.519130305381426343},
{11.65703219251637159757, 15.4312892102683783667, 18.9229991985461484779, 22.2953480191307663253, 25.6028559538106470696, 28.8703733470426581294, 32.1111962396826007673, 35.3331941827164603902, 38.5413648516782581263, 41.7390528671287491231, 44.9285896766809599002, 48.1116545549751500584, 51.2894900802831036514, 54.4630367429003871555, 57.6330202624747319113, 60.8000100548464695329, 63.9644594581674772787, 67.1267340677489918984, 70.2871321101665449958, 73.4458993623276834332, 76.6032402546531639901, 79.7593262554541405081, 82.914302285927626852, 86.0682916871853763418, 89.2214001081635776983, 92.3737185792999568267, 95.5253259648442163494, 98.6762909360144538935, 101.82667357108930624, 104.9765266624335123661, 108.1258967913838927085, 111.2748252178303175599, 114.4233486208057934287, 117.5714997184708738337, 120.7193077898484218776, 123.8667991160417497485, 127.0139973550964946297, 130.1609238618851917443, 133.3075979622130137721, 136.4540371886224707327, 139.6002574840085724529, 142.7462733780646477909, 145.8920981407024576296, 149.0377439158824198487, 152.183221838715302251, 155.3285421382282551, 158.473714227804235771, 161.618746784988038333, 164.763647822091146274, 167.908424748811084084, 171.053084427900588448, 174.197633224771148682, 177.342077051788978346, 180.486421407915002221, 183.630671414250515551, 186.77483184597398314, 189.918907161089699733, 193.062901526353850142, 196.206818840696342464, 199.350662756416365679, 202.49443669839488866, 205.638143881537394996, 208.781787326634307121, 211.92536987480418419, 215.06889420066537, 218.212362824364890517, 221.355778122578693397, 224.499142338584478614, 227.64245759149713089, 230.785725884746911191, 233.928949113871908919, 237.072129073688638861, 240.215267464897950906, 243.358365900177488875, 246.501425909806687048, 249.644448946865642563, 252.78743639204507409, 255.930389558100907688, 259.073309693983763514, 262.216197988670703773, 265.359055574724000761, 268.501883531599357507, 271.644682888723930251, 274.787454628362634062, 277.930199688289535682, 281.07291896427962989, 284.215613312434938458, 287.358283551357647556, 290.500930464181895805, 293.643554800474828099, 296.786157278016628639, 299.928738584468430134, 303.071299378936256207, 306.213840293438482518, 309.356361934283692281, 312.498864883365247322, 315.641349699378391109, 318.783816918965240399, 321.926267057792602858, 325.06870061156717527},
{12.79078171197211970881, 16.64100288151218877177, 20.1824707649491722718, 23.5912748179829668932, 26.9270407788180180777, 30.2172627093614007172, 33.476800819501472729, 36.7145291272447105052, 39.9361278108676769455, 43.1454250176031471788, 46.3451060653212837018, 49.5371160745469077376, 52.722901709733282573, 55.9035630241630818434, 59.0799524621934385072, 62.2527414665390427121, 65.4224665092662764596, 68.5895616504899422649, 71.7543820408036413559, 74.917221193885159314, 78.0783238852621290746, 81.2378959239360187497, 84.3961116514182644946, 87.5531197646858159941, 90.7090478863387722391, 93.8640061868419997658, 97.018090281466022864, 100.1713835665192626297, 103.3239591179727948687, 106.4758812455314482125, 109.6272067731887000884, 112.7779861009947800233, 115.9282640905649722945, 119.0780808076382712152, 122.2274721479731882902, 125.3764703674712425532, 128.5251045332402546751, 131.6734009090504525791, 134.8213832860767638054, 137.9690732677972761412, 141.1164905163084776835, 144.2636529660303354708, 147.4105770097384667658, 150.5572776610229798605, 153.703768696592712625, 156.8500627812875681226, 159.9961715782054816888, 163.1421058459746856948, 166.2878755248909416369, 169.4334898133810613868, 172.578957236038607235, 175.724285704297356366, 178.869482570656680808, 182.014554677245376415, 185.159508399402578812, 188.304349684862897426, 191.44908408905505779, 194.593716806956934203, 197.738252701893041924, 200.88269633161182276, 204.027051971938142754, 207.171323638260280501, 210.315515105079447821, 213.459629923822822348, 216.603671439097572603, 219.747642803542906138, 222.891546991419336781, 226.035386811058780963, 229.179164916285444911, 232.322883816905489686, 235.466545888352934728, 238.610153380569990407, 241.75370842619183158, 244.897213048098596332, 248.040669166390995475, 251.184078604840244702, 254.327443096857992251, 257.470764291027432677, 260.614043756232804173, 263.757282986420904306, 266.900483405025075991, 270.043646369079267538, 273.186773173047219053, 276.329865052389538273, 279.472923186889372586, 282.615948703755534352, 285.75894268052027095, 288.901906147747369066, 292.044840091564926944, 295.18774545603590293, 298.330623145378439991, 301.473474026046961581, 304.61629892868412362, 307.759098649952880693, 310.901873954257173459, 314.044625575359060988, 317.187354217899499669, 320.330060558829403348, 323.472745248757102129, 326.615408913217844974},
{13.91582261050489645303, 17.83864319920532394797, 21.42848697211535907, 24.8732139238751459062, 28.2371343599680992505, 31.5501883818318502328, 34.8286965376857087646, 38.0824790873276610617, 41.3178646902445384162, 44.5391446334094879773, 47.7493457344425978298, 50.950671453544500199, 54.1447680562547476183, 57.3328925814208656259, 60.516022838023195342, 63.6949317158587355364, 66.8702387400877754102, 70.0424466703442478037, 73.2119680107542338865, 76.3791445561388427882, 79.5442620333222772758, 82.7075612248870908977, 85.8692465291850706398, 89.0294926243038924299, 92.1884497110917729338, 95.3462476783460860806, 98.5029994413218294332, 101.6588036397013767962, 104.8137468345654282716, 107.9679053100788411759, 111.1213465607647771847, 114.2741305268015084244, 117.4263106259508628458, 120.5779346202638827143, 123.7290453477207746381, 126.879681342813101663, 130.029877365306343691, 133.1796648526941633101, 136.3290723089239339698, 139.4781256396518053435, 142.6268484424364238526, 145.7752622587988283971, 148.9233867938824105649, 152.0712401084801665861, 155.218838787409721518, 158.3661980875732528372, 161.5133320685109661259, 164.6602537078207915048, 167.8069750034558031494, 170.9535070646104936685, 174.0998601926562803244, 177.2460439533765251103, 180.3920672415747089199, 183.5379383389803873587, 186.6836649662514458253, 189.8292543297641135954, 192.974713163791041315, 196.120047768589908222, 199.265264044858374693, 202.41036752495397179, 205.555363401228262051, 208.700256551782100076, 211.845051563912053336, 214.989752755486164514, 218.134364194459533938, 221.278889716716074497, 224.423332942401735526, 227.567697290896078064, 230.71198599455294798, 233.85620211132682657, 237.000348536388976743, 240.144428012826521456, 243.288443141507896812, 246.432396390189549455, 249.576290101931154243, 252.720126502879890808, 255.863907709478330457, 259.007635735145156293, 262.151312496473189682, 265.294939818984956344, 268.438519442482235181, 271.582053026022640066, 274.72554215255324341, 277.868988333228520504, 281.012393011437440083, 284.155757566562318799, 287.299083317490068007, 290.442371525894666892, 293.585623399308075305, 296.72884009399533442, 299.872022717648276898, 303.015172331911066339, 306.15828995474969544, 309.301376562676582119, 312.444433092840502782, 315.587460444991282858, 318.730459483327918686, 321.873431038238124668, 325.016375907936678845, 328.159294860009373083},
{15.03346930374343806413, 19.02585353612775990436, 22.662720658136055604, 26.1427676433791000096, 29.5346341078439244295, 32.8705345976875351875, 36.1681571359112274793, 39.4382144800080578106, 42.6876512846610891312, 45.9212017638355864336, 49.1422214247461477537, 52.3531638708113882714, 55.555869720665701492, 58.7517496714094626624, 61.9419048384612890354, 65.1272083474595863049, 68.3083621338320306369, 71.4859373958292209502, 74.6604039857647266985, 77.8321521433816273377, 81.0015088201722864313, 84.1687501140817867501, 87.3341108619538513306, 90.4977921247575201122, 93.6599670898802417169, 96.82078576999569792, 99.9803787769322710628, 103.1388603773396636799, 106.2963309854999606644, 109.4528792112064413934, 112.6085835530974945588, 115.7635138073537304384, 118.9177322462829347183, 122.071294609654084653, 125.2242509427215879698, 128.3766463080032344638, 131.5285213925317813745, 134.6799130281184966843, 137.8308546388724419434, 140.981376627606952671, 144.1315067106808212104, 147.2812702091496711293, 150.4306903027541820243, 153.5797882521780651262, 156.7285835941173894942, 159.8770943129731240927, 163.0253369923786248919, 166.173326949278127151, 169.3210783528612391376, 172.468604330316196833, 175.6159170610786350702, 178.7630278610127584764, 181.9099472577599188764, 185.056685058319141681, 188.2032504097797446444, 191.3496518540034968213, 194.4958973769492029655, 197.6419944532432350186, 200.7879500865229484127, 203.9337708460141148826, 207.079462899746813641, 210.225032044765265304, 213.370483734644711982, 216.51582310459167405, 219.661054994371935716, 222.806183969282742074, 225.951214339361350837, 229.09615017700078194, 232.240995333124931206, 235.385753452058805915, 238.530427985215197867, 241.675022203706374876, 244.819539209978122927, 247.963981948553517998, 251.108353215964985084, 254.252655669945370847, 257.396891837941793415, 260.541064125009832462, 263.685174821140093089, 266.829226108064238349, 269.973220065583168236, 273.117158677456066974, 276.261043836885492562, 279.404877351630496535, 282.548660948776897184, 285.692396279191250358, 288.836084921682737239, 291.979728386895090456, 295.123328120948784274, 298.266885508851999958, 301.410401877697324629, 304.553878499659734323, 307.697316594810134485, 310.84071733375757063, 313.984081840132166509, 317.127411192919886441, 320.270706428659343058, 323.413968543510073328, 326.557198495200976816, 329.700397204866944118},
{16.14474294230134099874, 20.2039426328117258759, 23.886530755968385445, 27.4012592588663382431, 30.8207940864510095666, 34.1794746664832437846, 37.4962736357858113951, 40.7827470981251237079, 44.0464252109438020911, 47.2924656052694268289, 50.5245397255712282143, 53.7453428657930609587, 56.9569043527017959786, 60.1607847679751571146, 63.3582060271058104104, 66.5501398541523807407, 69.737369558625451747, 72.9205341597059750703, 76.1001605323206286403, 79.2766872393004270287, 82.4504824763661549833, 85.6218577734820543393, 88.7910785880958353643, 91.958372588947481359, 95.1239362013536302566, 98.2879398280812180259, 101.4505320502336470142, 104.6118430346937895829, 107.7719873186217754541, 110.9310661006658000942, 114.0891691384409693642, 117.246376329404138047, 120.4027590353780662156, 123.5583811981635579221, 126.7133002838623975037, 129.8675680859540629621, 133.0212314112714208938, 136.1743326683987345199, 139.3269103743688571142, 142.4789995926412357814, 145.6306323130296903698, 148.7818377823910733491, 151.9326427933853262654, 155.0830719373991791877, 158.2331478267318269725, 161.3828912903262555865, 164.5323215466591803049, 167.6814563568480104628, 170.8303121605728884779, 173.9789041970281814618, 177.1272466127968615526, 180.2753525582717805696, 183.4232342740208816569, 186.5709031683015710795, 189.7183698867668416364, 192.8656443752674333383, 196.012735936536352988, 199.1596532814411754449, 202.3064045754030113474, 205.4529974805066094236, 208.5994391937619056199, 211.7457364819218854019, 214.8918957132135933507, 218.0379228862974186685, 221.1838236567334924888, 224.3296033612023823958, 227.4752670396996098664, 230.620819455899295319, 233.766265115860983948, 236.911608285235023936, 240.056853005105415813, 243.202003106594535742, 246.347062224341305974, 249.492033808953027464, 252.636921138521017345, 255.781727329281247678, 258.926455345493223335, 262.071108008603246183, 265.215688005751885966, 268.360197897679824885, 271.504640126081184194, 274.649017020448908241, 277.793330804452714359, 280.937583601886463047, 284.081777442218515974, 287.225914265775688867, 290.369995928588736622, 293.514024206924897707, 296.658000801530846201, 299.801927341607428338, 302.945805388535774347, 306.089636439372756633, 309.233421930132294802, 312.377163238867671607, 315.52086168856880812, 318.664518549887339279, 321.80813504370132181, 324.951712343530485764, 328.095251577812100032, 331.238753832046753615},
{17.25045478412596540986, 21.37397218116274843683, 25.1010385210080587488, 28.6497962617288667042, 32.0966767253863573467, 35.4780131751778988777, 38.8139888812133163881, 42.1169585325496030085, 45.3950094398602866357, 48.6537041118083899389, 51.89701752445788318, 55.1278782243997677791, 58.3484984443126675298, 61.5605846363496284618, 64.7654767358390302248, 67.9642431480985027183, 71.1577472498198409683, 74.3466950092901025645, 77.5316697642388281109, 80.7131580653214383525, 83.8915691789826263424, 87.0672500100105533532, 90.2404966624437837542, 93.4115634976832588552, 96.5806703049001313872, 99.7480080307297377318, 102.913743397433539272, 106.0780226549243369374, 109.2409746516512198465, 112.4027133652576467654, 115.5633400013805598432, 118.7229447446744239588, 121.8816082278448046627, 125.0394027705592692744, 128.1963934294281725076, 131.3526388919930689419, 134.5081922412289116413, 137.6631016120194282529, 140.8174107570783382314, 143.9711595366197594724, 147.1243843435465413501, 150.2771184738867283054, 153.4293924505599731407, 156.581234307215987046, 159.7326698377929350364, 162.8837228165459066802, 166.0344151925556989178, 169.1847672621157963953, 172.3347978218865914303, 175.4845243052814187614, 178.6339629041935864319, 181.7831286778749661387, 184.932035650524938388, 188.0806968989355242737, 191.2291246313578284983, 194.3773302586011171016, 197.5253244582445701766, 200.6731172327293727178, 203.8207179620023521817, 206.9681354512993666799, 210.1153779745850370176, 213.2624533141034841593, 216.4093687964410465018, 219.5561313254553047756, 222.7027474123841235827, 225.8492232034129728899, 228.9955645049477972411, 232.1417768068135418214, 235.2878653035746014168, 238.4338349141524863865, 241.5796902998975193572, 244.7254358812550623961, 247.8710758531523479707, 251.016614199219208166, 254.162054704944658672, 257.307400969861218456, 260.452656418839878358, 263.597824312570638163, 266.742907757296395769, 269.887909713861592701, 273.032833006131309154, 276.177680328831382161, 279.322454254855524801, 282.467157242081293826, 285.611791639733035749, 288.756359694326591947, 291.900863555227521553, 295.045305279851871647, 298.189686838536056282, 301.334010119100171697, 304.478276931127050331, 307.622489009977519255, 310.766648020560660368, 313.910755560876352923, 317.054813165345998668, 320.198822307946072507, 323.342784405157995017, 326.486700818746776772, 329.630572858379928436, 332.774401784097256494},
{18.35126149594727247464, 22.53681707111983929192, 26.30718156121711780949, 29.8893163210056282598, 33.3631912390186349899, 36.7670179392397985337, 40.1221241286456884613, 43.4416223752821230302, 46.734130921754099992, 50.0055997010855610101, 53.2602953222855027155, 56.501371328065474847, 59.7312170537446423488, 62.9516807071213473915, 66.1642173111047364013, 69.3699898602310758627, 72.5699403103336189366, 75.7648405290955255843, 78.9553295878297598432, 82.1419415314487202478, 85.3251263774836449634, 88.5052662145282774035, 91.6826876972908137039, 94.8576718540456785579, 98.0304618634731948263, 101.2012692791168100619, 104.3702790542140442295, 107.5376536302828543532, 110.7035362883217845679, 113.8680539143201063411, 117.0313192959057824914, 120.193433040905667023, 123.3544851889330901863, 126.5145565721470920978, 129.6737199698290023981, 132.832041092519409888, 135.9895794245137458822, 139.1463889480587095842, 142.3025187682767576311, 145.4580136544117124383, 148.6129145102390934983, 151.7672587842710774127, 154.9210808285940930271, 158.0744122137191988861, 161.2272820056335526471, 164.3797170102624262622, 167.5317419897437473848, 170.6833798542482500623, 173.8346518325219877259, 176.9855776238634757912, 180.1361755338584987739, 183.2864625958682936223, 186.436454679990626986, 189.5861665909794739881, 192.7356121564104561053, 195.8848043062100806946, 199.0337551445223631228, 202.182476014762665526, 205.330977558602297987, 208.4792697695358978435, 211.6273620416045838253, 214.7752632137794981598, 217.9229816104510314004, 221.0705250784174530542, 224.2179010207217359514, 227.3651164276461264024, 230.5121779051396820639, 233.6590917009239016777, 236.8058637284951325363, 239.9524995892191740348, 243.0990045926929813277, 246.2453837755302540625, 249.3916419187116641447, 252.5377835636262691159, 255.683813026918046028, 258.8297344142402676951, 261.9755516330104584092, 265.1212684042497617149, 268.2668882735826001466, 271.412414621465394079, 274.557850672706736468, 277.703199505335707042, 280.848464058869878899, 283.99364714202995708, 287.138751439943835211, 290.283779520879112233, 293.428733842539732221, 296.573616757959357127, 299.718430521021320377, 302.863177291632508072, 306.007859140576246945, 309.152478054067220209, 312.297035938029562695, 315.441534622117586238, 318.585975863497038216, 321.730361350403385513, 324.874692705492329442, 328.018971488996582446, 331.163199201701863809, 334.307377287754089733},
{19.44770310809473507372, 23.69320803747137816335, 27.50575308084526679401, 31.1206214136035017476, 34.6211227000072361311, 38.0472445886101995713, 41.4213998124606061373, 44.7574217886310552072, 48.0644355001732959175, 51.3487619649376744423, 54.6149481140489624312, 57.8663645101606108733, 61.1055718873714344585, 64.3345560878828850118, 67.5548842230513459128, 70.7678116849179104419, 73.9743574048928428322, 77.175357979301749061, 80.3715073634700602983, 83.5633864869637938315, 86.7514856890371562012, 89.9362219486252595508, 93.1179522803525728635, 96.2969842663050507251, 99.4735844203237424028, 102.6479848927540357414, 105.8203888908621466353, 108.9909750954635916493, 112.1599012858728179, 115.3273073351958564328, 118.4933177009093243784, 121.6580435079294646399, 124.8215843004175689902, 127.9840295225895758595, 131.1454597765107227497, 134.305947895332212475, 137.4655598629890797064, 140.6243556055287924283, 143.7823896746088507951, 146.9397118400119363852, 150.0963676050701220839, 153.2523986565064522181, 156.4078432582712308166, 159.5627365973778656469, 162.7171110884563787679, 165.8709966426849464111, 169.0244209058865115269, 172.1774094698534483746, 175.3299860603605412483, 178.4821727048229128025, 181.6339898821331682398, 184.7854566568565549075, 187.9365907996627667314, 191.0874088956187265716, 194.2379264417505756061, 197.3881579350989011255, 200.5381169523337791141, 203.6878162218612390289, 206.8372676892367540217, 209.986482576601397897, 213.1354714367699567969, 216.2842442025255063636, 219.432810231610061086, 222.5811783478444410029, 225.7293568787612764614, 228.8773536900920658509, 232.0251762174115515039, 235.1728314952096527607, 238.3203261836321688974, 241.4676665931059060935, 244.6148587070413366959, 247.7619082027859755609, 250.908820470984022017, 254.0556006334821784087, 257.2022535599076682862, 260.3487838830321238845, 263.495196013024007188, 266.6414941506824098351, 269.7876822997363037789, 272.933764278285464476, 276.0797437294522546071, 279.2256241313071455132, 282.3714088061251839437, 285.5171009290255116451, 288.6627035360414518349, 291.808219531664534327, 294.953651695902091222, 298.099002690884674609, 301.244275067056488665, 304.389471268979257169, 307.534593640777434073, 310.679644431250382883, 313.824625798675076722, 316.969539815320984261, 320.114388471697088511, 323.25917368054941924, 326.403897280626050782, 329.548561040225211782, 332.693166660540960473, 335.837715778819787442},
{20.54022982504821067967, 24.84376259758634850946, 28.69743099361502567178, 32.3444036366282440702, 35.8711544006491411375, 39.3193557607707633204, 42.7124519715173083417, 46.0649635693805647717, 49.3864999788272321709, 52.683738054305582644, 55.9614943577656875184, 59.2233488311465913034, 62.4720280658788344369, 65.7096514739012568295, 68.9378952482337413008, 72.1581049409241768561, 75.3713747879784513776, 78.5786048650294764217, 81.7805430787192800167, 84.9778165501386843426, 88.1709554295107574055, 91.3604112148814694887, 94.5465710167049096891, 97.7297687892847055982, 100.9102942636303734534, 104.0884001179375929891, 107.2643077823104397531, 110.4382121746562788078, 113.6102855925326607103, 116.780680932856428029, 119.949534372202061728, 123.1169676110688148055, 126.2830897632994143536, 129.4479989548911133741, 132.6117836833981843659, 135.7745239790053988634, 138.9362924004410119235, 142.0971548926696814707, 145.2571715283700321022, 148.4163971512655419864, 151.5748819362200225997, 154.7326718784619450707, 157.8898092222363156905, 161.0463328374992443308, 164.2022785518915475434, 167.3576794440933500757, 170.5125661037242934328, 173.6669668621761825682, 176.8209079981169661561, 179.9744139208631207905, 183.1275073343627648404, 186.28020938414883363, 189.4325397892980008049, 192.5845169611566535204, 195.7361581103619004011, 198.8874793434865785661, 202.0384957504669953697, 205.1892214838261266955, 208.3396698305794176979, 211.4898532776020552346, 214.6397835711429888568, 217.7894717710898835048, 220.938928300518759871, 224.0881629910007774677, 227.2371851240851422378, 230.3860034693303774104, 233.5346263192152528963, 236.683061521224734368, 239.8313165073747149604, 242.9793983214114553931, 246.1273136438970902996, 249.2750688153708376964, 252.422669857756312161, 255.5701224941682779939, 258.7174321672570167718, 261.8646040562149910338, 265.0116430925584603043, 268.1585539747859717735, 271.3053411820060531487, 274.4520089866178464613, 277.5985614661207229267, 280.7450025141220088002, 283.8913358506057417013, 287.0375650315197881069, 290.1836934577336171572, 293.3297243834144832946, 296.4756609238656674505, 299.6215060628667166298, 302.767262659552263376, 305.9129334548629630746, 309.0585210775993258538, 312.204028050106712132, 315.349456793617481019, 318.494809633274205985, 321.640088802855982149, 324.785296449228126122, 327.93043463653399633, 331.075505350146224715, 334.220510500393336708, 337.365451926076533765},
{21.62922143659035612018, 25.98900797641358678057, 29.88279992365850541532, 33.5612650491746761404, 37.1138853016453953094, 40.5839362781940992378, 43.9958453948690754332, 47.3647895256857827307, 50.7008419785362646003, 54.0110212378794548932, 57.3004034264614452438, 60.572770587121562543, 63.8310098273251325132, 67.0773701619023361795, 70.3136338919326718246, 73.5412344846535055979, 76.7613397775037400704, 79.9749120292460000811, 83.1827521103117237367, 86.385532581722497923, 89.5838228346359335388, 92.7781084566773907335, 95.9688063338303773883, 99.1562765575446810004, 102.3408319076056624461, 105.5227454739223503131, 108.7022568342801993832, 111.8795771006461595378, 115.054893070927699426, 118.2283706675693059819, 121.4001578031804679508, 124.5703867825071971084, 127.7391763266781496482, 130.9066332877927958285, 134.0728541081545354865, 137.2379260677609895777, 140.4019283552983502024, 143.5649329912948482512, 146.7270056268597085359, 149.8882062372606157789, 153.0485897262420376734, 156.2082064542815378642, 159.3671027017855455713, 162.5253210764349824646, 165.6829008724231312242, 168.8398783881194117647, 171.9962872076931850582, 175.1521584514017038489, 178.3075209985543831189, 181.4624016865865203523, 184.6168254891892740269, 187.7708156760328387418, 190.9243939562731643127, 194.0775806077385440564, 197.2303945934422016584, 200.3828536668534720975, 203.5349744671774072979, 206.6867726057357772753, 209.8382627444074535299, 212.9894586669697050488, 216.1403733440812170579, 219.2910189925603242074, 222.4414071295360751566, 225.591548621983667506, 228.7414537320981213817, 231.8911321589096216475, 235.040593076499756226, 238.1898451691390642426, 241.3388966636321607071, 244.4877553591266075242, 247.6364286546151257334, 250.7849235743372379197, 253.9332467912656030357, 257.0814046488438226299, 260.2294031811260676916, 263.3772481314542480871, 266.5249449697954043325, 269.6724989088503552961, 272.8199149190342209184, 275.9671977424201128734, 279.1143519057289231047, 282.2613817324406308873, 285.4082913540957975703, 288.5550847208498402715, 291.7017656113371975426, 294.8483376418975562652, 297.9948042752118423677, 301.1411688283916376633, 304.2874344805620263871, 307.4336042799745581662, 310.579681150684004006, 313.7256678988198471836, 316.8715672184809639383, 320.0173816972796848205, 323.1631138215593644877, 326.3087659813077059601, 329.4543404747863673149, 332.59983951289580881, 335.745265223292902423, 338.890619654277511163},
{22.71500167497224254021, 27.1293984123007411125, 31.06236810514955592159, 34.77173314673616698973, 38.3498438244546090971, 41.8415052903780148608, 45.2720842527724307531, 48.6573857681979493525, 52.0079280591701441013, 55.3310580119999005162, 58.6321018452644938918, 61.9150367950023815152, 65.1829053642531854298, 68.4380823278079186558, 71.6824531820840231906, 74.9175370846151118142, 78.1445737646462201753, 81.3645863447466025106, 84.5784276382892187732, 87.7868148555074169179, 90.9903560164616591344, 94.1895703259092975885, 97.3849040825137785604, 100.5767432384432452599, 103.7654234142139802644, 106.9512379576577147575, 110.1344444835765087245, 113.3152702216485391168, 116.4939164210942395589, 119.6705620025679139135, 122.8453666046353676037, 126.0184731398500947358, 129.1900099509270441368, 132.3600926387671139892, 135.5288256196282764559, 138.6963034575007127477, 141.862612008941722166, 145.0278294106846978509, 148.192026934825853123, 151.3552697319904298868, 154.5176174793430721563, 157.6791249474491224757, 160.8398424976721986126, 163.9998165198983463643, 167.159089818822659547, 170.3177019557534806136, 173.4756895518294060553, 176.6330865576635980849, 179.7899244936951698277, 182.946232664912149784, 186.1020383530934508889, 189.2573669892811951517, 192.4122423088257620534, 195.5666864910327173837, 198.7207202851740924228, 201.8743631243987394337, 205.0276332288814483761, 208.1805476993830177748, 211.3331226022492584205, 214.4853730467524203528, 217.6373132555708086683, 220.7889566291089131063, 223.9403158042791396651, 227.0914027082954487448, 230.24222860796739575, 233.3928041549289862112, 236.5431394271893373243, 239.6932439673504789312, 242.84312681780095993, 245.9927965531615933034, 249.1422613102311110528, 252.2915288156542257271, 255.4406064115121918109, 258.5895010790160692624, 261.738219460465203007, 264.8867678796176779094, 268.0351523606054559512, 271.1833786455143488371, 274.3314522107377489986, 277.4793782822029814573, 280.6271618495601141377, 283.7748076794149576298, 286.9223203276806942331, 290.0697041511160100627, 293.2169633181116836632, 296.3641018187822400265, 299.5111234744144484876, 302.6580319463200721957, 305.8048307441363175596, 308.9515232336138415767, 312.0981126439289154188, 315.2446020745533804737, 318.3909945017133383217, 321.5372927844650622703, 324.6834996704143813423, 327.8296178011037468446, 330.9756497170893279193, 334.1215978627287788606, 337.2674645906987623159, 340.4132521662598851798},
{23.79784903412830132652, 28.26532843664237089487, 32.23658055304289543972, 35.97627308669031095202, 39.5794988808720523338, 43.0925260900467780489, 46.5416207763904627615, 49.943190364080539447, 53.3081804654885126803, 56.6442540504621828341, 59.9569785470631804873, 63.2505198442508703826, 66.5280709502253927965, 69.7921286962553809626, 73.0446789405619107544, 76.2873243450505657953, 79.5213748333260270602, 82.7479130669314924482, 85.9678427643474011684, 89.1819249701019659438, 92.3908056926507010381, 95.5950372506966690379, 98.7950949611096972814, 101.9913903287711443415, 105.1842815759721012439, 108.3740821248311499475, 111.5610674879686050119, 114.7454809093412447015, 117.9275380148731077462, 121.1074306720607649959, 124.2853302127964021358, 127.4613901399016353124, 130.635748412267219514, 133.808529383902568964, 136.979845457077448955, 140.1497984979738061304, 143.3184810540449429399, 146.4859774050015396897, 149.6523644735620650084, 152.8177126174829195945, 155.9820863206668878392, 159.1455447981431451952, 162.3081425272691932968, 165.4699297155094665969, 168.6309527135073315842, 171.7912543808166095489, 174.9508744105403491842, 178.109849618194622359, 181.2682141993388152969, 184.4259999598634094687, 187.5832365222792320209, 190.7399515108905573198, 193.8961707183436293309, 197.0519182557102399724, 200.2072166879832081111, 203.3620871566189721239, 206.5165494905554602239, 209.670622306955502959, 212.8243231027727953238, 215.9776683381050529092, 219.1306735121844086028, 222.2833532327556441792, 225.4357212795063453329, 228.5877906621376573474, 231.739573673598437875, 234.8910819389479292833, 238.042326460261481861, 241.1933176579493945382, 244.3440654088197858633, 247.494579081181867245, 250.6448675672554651213, 253.7949393131256152943, 256.94480234645708923, 260.0944643021624274392, 263.2439324461981196531, 266.3932136976467000484, 269.5423146492274706665, 272.6912415863651121978, 275.840000504933401022, 278.9885971277804612794, 282.1370369201322978941, 285.2853251039626557691, 288.433466671409422124, 291.5814663973107368172, 294.7293288509276147339, 297.8770584069141402849, 301.0246592555911008728, 304.1721354125742256616, 307.3194907278039366367, 310.46672889401965514, 313.6138534547181983525, 316.7608678116326105941, 319.9077752317648718535, 323.0545788540032821512, 326.2012816953529097728, 329.3478866568052914156, 332.4943965288715625793, 335.6408139968013589769, 338.7871416455081480647, 341.9333819642201094382},
{24.8780050582071901033, 29.39714321344088307415, 33.40582946173510907262, 37.17529746012897103684, 40.8032687865001493982, 44.337414125162324138, 47.804862406319513403, 51.2225996928570254586, 54.6019827712596995574, 57.9509792172987411245, 61.2753893289661666758, 64.5795614647701075505, 67.8668344792526446976, 71.1398237042991529678, 74.400612617155900559, 77.6508852501824312097, 80.8920200494474391792, 84.1251578981747482596, 87.3512523774997613482, 90.5711075386380213106, 93.7854067200337236568, 96.9947348300546205631, 100.1995957864242634271, 103.4004263150553231949, 106.5976069772670663773, 109.7914710623922431376, 112.9823118189003158615, 116.1703883796906344974, 119.3559306518672346307, 122.5391433785415071255, 125.7202095335217084592, 128.899293174654570454, 132.0765418549472511, 135.2520886701957601092, 138.4260540060867587955, 141.5985470354700694265, 144.7696670068755058081, 147.9395043577464253232, 151.1081416798191413657, 154.2756545592428823056, 157.4421123101448644881, 160.6075786171974272989, 163.7721121001839389558, 166.9357668114673095343, 170.0985926755460021795, 173.2606358784642246047, 176.4219392136678544796, 179.5825423899199368784, 182.7424823060729302026, 185.9017932968101766157, 189.0605073528929458671, 192.2186543189629597154, 195.3762620715382283493, 198.5333566794898429243, 201.6899625489888665554, 204.8461025546572745489, 208.0017981584381129814, 211.1570695175119676158, 214.3119355824247292024, 217.4664141864515664978, 220.6205221271006849238, 223.7742752405551011186, 226.9276884697589869055, 230.0807759267751802908, 233.2335509499705786133, 236.3860261565249244719, 239.5382134907047835525, 242.6901242682972860985, 245.8417692175565995399, 248.9931585169793822286, 252.1443018301930058643, 255.2952083382115844671, 258.4458867692893434154, 261.5963454265781999728, 264.7465922137762582475, 267.8966346589359446607, 271.0464799365844625714, 274.1961348882948987483, 277.345606041833471252, 280.4948996289968946558, 283.6440216022435028324, 286.7929776502124790441, 289.9417732122171813878, 293.0904134917910164773, 296.2389034693575156041, 299.3872479140901261906, 302.5354513950216771426, 305.6835182914584478574, 308.8314528027492123892, 311.9792589574554940971, 315.1269406219655089161, 318.2745015085908589391, 321.4219451831819282102, 324.5692750722950991692, 327.7164944699423239861, 330.8636065439512259081, 334.0106143419617500983, 337.157520797083411924, 340.3043287332353859129, 343.45104087019002508},
{25.95568078504013698193, 30.52514669524632025273, 34.5704625115363524631, 38.36917418523642855628, 42.0215285333218034213, 45.5765435956207278963, 49.0621777272466022097, 52.4959737631437846456, 55.88968463366603656, 59.2515718166494818237, 62.5876606531293668062, 65.9024761298616912173, 69.1994985170141734891, 72.4814582419270302496, 75.7505337555910370362, 79.0084883875362540938, 82.2567674693771693769, 85.4965688058304229435, 88.7288948029318385356, 91.9545916871332518483, 95.1743794588275520031, 98.3888750783079994298, 101.5986106309231152384, 104.8040477146469270651, 108.0055889490973004766, 111.2035872655783314552, 114.3983534684376182792, 117.5901624365918375127, 120.7792582457883995603, 123.9658584271937836133, 127.1501575295321185272, 130.3323301156154372924, 133.5125332964726467638, 136.6909088851034939204, 139.8675852355118510304, 143.042678819917353184, 146.2162955870329869155, 149.388532136383535321, 152.5594767373447609879, 155.729210216544096664, 158.8978067332062294953, 162.0653344587419438506, 165.2318561742049346739, 168.3974297970543893885, 171.5621088468638737455, 174.7259428581333574349, 177.8889777471311067338, 181.0512561386681270289, 184.2128176578519976422, 187.3736991911489960203, 190.5339351204789790686, 193.6935575335568782162, 196.8525964132618867317, 200.0110798084474470867, 203.1690339882913260277, 206.3264835820166433719, 209.4834517055844831641, 212.6399600767606949605, 215.7960291197887352159, 218.9516780607527790083, 222.1069250145874045555, 225.2617870645790295227, 228.4162803351075321085, 231.5704200582920759144, 234.7242206351313469577, 237.8776956916637390035, 241.0308581306162451146, 244.1837201789608689725, 247.3362934317533541146, 250.4885888925901683173, 253.6406170109853096601, 256.7923877169380493277, 259.9439104529357020541, 263.0951942036114936509, 266.246247523256209006, 269.3970785613632349477, 272.5476950863695840302, 275.6981045077402571226, 278.8483138965296651057, 281.9983300045416000045, 285.14815928219826349, 288.2978078952189856782, 291.4472817402003765221, 294.5965864591816377336, 297.7457274532715299168, 300.8947098954069535644, 304.0435387423071897012, 307.1922187456824907735, 310.3407544627508569135, 313.4891502661124255555, 316.637410353026897917, 319.7855387541357835221, 322.9335393416679276374, 326.0814158371637640341, 329.2291718187509781807, 332.3768107280017481544, 335.5243358763994293081, 338.6717504514404434885, 341.8190575223952059721, 344.9662600457501567316},
{27.03106182135001690378, 31.64960813250952753606, 35.73078957439769855586, 39.55823294471697132515, 43.2346157755107242762, 46.8102529282513922792, 50.3139014311986569771, 53.7636406903962657133, 57.1716058234388108539, 60.546342216858952611, 63.894092906009007105, 67.2195539894612843505, 70.5263429436668280014, 73.8173020365337876355, 77.0947021483086511927, 80.3603838984355963895, 83.6158579086068175275, 86.8623776288645693888, 90.1009932640510500622, 93.3325923873310896114, 96.5579309898147290476, 99.7776575385633297809, 102.9923318430519468914, 106.2024400122256825262, 109.4084064300284764675, 112.6106034306555252763, 115.8093591803106852893, 119.0049641470107847484, 122.1976764488689455705, 125.3877263041892082081, 128.5753197567250373616, 131.7606418118346424101, 134.9438590906732689684, 138.1251220876332636289, 141.3045670992813723762, 144.4823178798199644093, 147.6584870677140740065, 150.8331774199133242067, 154.0064828835600302178, 157.178489529838218831, 160.3492763703994377242, 163.5189160733834732692, 166.6874755932688080624, 169.8550167265097665525, 173.0215966030442252925, 176.1872681222086281259, 179.3520803403136461775, 182.5160788160648409837, 185.679305919118821208, 188.8418011063151479005, 192.0036011694923061166, 195.1647404582619424027, 198.3252510806626411762, 201.4851630842292230865, 204.6445046196847998877, 207.8033020891814969576, 210.9615802807743384863, 214.119362490605059895, 217.2766706340934007267, 220.4335253472784347426, 223.5899460793181148359, 226.7459511770384328806, 229.9015579623218831561, 233.0567828030361345738, 236.2116411781261539322, 239.3661477374249474438, 242.5203163566782983601, 245.6741601882262649422, 248.8276917077378174534, 251.980922757354023853, 255.1338645855589452014, 258.2865278840652747123, 261.4389228219732329522, 264.5910590774358703162, 267.7429458670413410495, 270.8945919731025664738, 274.0460057690267083339, 277.197195242920773622, 280.3481680195752495472, 283.498931380954729086, 286.6494922853128654462, 289.7998573850385402858, 292.9500330433307161691, 296.1000253497909546887, 299.2498401350149177012, 302.3994829842572418395, 305.548959250237907817, 308.6982740651525470066, 311.847432351943977217, 314.9964388348875832942, 318.1452980495389079901, 321.2940143520879516614, 324.4425919281611576641, 327.5910348011088498235, 330.7393468398129587484, 333.8875317660471979306, 337.0355931614194042701, 340.1835344739235191868, 343.3313590241266363399, 346.4790700110146626966},
{28.10431238769707462666, 32.77076732419575932909, 36.88708817965733110202, 40.74277048198659875051, 44.44283579577298211625, 48.0388493544156290899, 51.5603384966502464118, 55.0259004918103609506, 58.4480396610140842845, 61.8355759577394747758, 65.1949632073770184155, 68.5310634104732623956, 71.8476272530765599144, 75.1476057362607086912, 78.4333597266773230354, 81.7068051954773658257, 84.969516504672402141, 88.2228015023777372327, 91.467757182950829376, 94.7053116458208751606, 97.9362562033929690332, 101.1612702816872048415, 104.3809409650203530583, 107.5957785042766088716, 110.8062287443798924847, 114.0126831730652063465, 117.2154871135961108634, 120.4149464551830536772, 123.6113332210335434025, 126.8048902048250899261, 129.9958348548615655587, 133.1843625463637719401, 136.3706493528301013987, 139.5548544047532720622, 142.7371219064502182118, 145.9175828680890703464, 149.0963565992524773217, 152.2735520018742467562, 155.4492686936142915089, 158.6235979873096827909, 161.7966237477647628437, 164.9684231435972214944, 168.1390673089677249772, 171.3086219276547729736, 174.4771477499899839018, 177.6447010515604107707, 180.8113340412494251397, 183.977095225075133605, 187.1420297313545116666, 190.3061796019398422501, 193.4695840536153751844, 196.6322797131851458342, 199.7943008293103589415, 202.9556794637525907345, 206.1164456643357697053, 209.2766276216459957781, 212.4362518112359373215, 215.5953431228833381833, 218.7539249782656998369, 221.9120194382509987669, 225.0696473008636122565, 228.2268281908623167634, 231.3835806417606509704, 234.5399221710268725618, 237.6958693491192994945, 240.8514378629414149573, 244.0066425742383716176, 247.1614975734012970137, 250.3160162290970885274, 253.4702112340983458333, 256.6240946476499936417, 259.7776779346753722063, 262.9309720020945779947, 266.0839872325011561354, 269.236733515419476372, 272.3892202763439154426, 275.5414565037420170615, 278.6934507741868413687, 281.8452112757685180321, 284.9967458299213794997, 288.1480619117907967374, 291.2991666692528144723, 294.4500669406897506717, 297.6007692716159662768, 300.7512799302399208972, 303.9016049220413154426, 307.0517500034355015679, 310.2017206945913378043, 313.3515222914632291154, 316.5011598770931433457, 319.6506383322339040186, 322.7999623453409691454, 325.9491364219761800409, 329.0981648936635668409, 332.2470519262341965671, 335.3958015276942166405, 338.5444175556476561829, 341.6929037243031763867, 344.8412636110917891478, 347.9895006629205716407},
{29.17557857691904388275, 33.88883889413516911341, 38.03960800837917438035, 41.92305499356922049302, 45.6464656563403003783, 49.2626127633955018893, 52.8017677295646841512, 56.2830283213221614436, 59.7192559621920005965, 63.1195364282239512927, 66.490527842779243065, 69.8372531866111761167, 73.1635925614867209654, 76.4726027373776318753, 79.7667322252544443226, 83.0479704801092365469, 86.3179541028276282399, 89.5780441246123888334, 92.8293833405810302185, 96.0729395679264797815, 99.309538776595255159, 102.5398908048524341027, 105.7646095603629471213, 108.9842290623515606681, 112.1992163071599346672, 115.4099816793763766244, 118.6168874464707201645, 121.8202547424590508662, 125.0203693496871830496, 128.2174865167176877341, 131.411834997282060446, 134.6036204553034489907, 137.7930283505921158721, 140.9802263964715045256, 144.1653666625167312182, 147.3485873814798232592, 150.5300145083841189674, 153.7097630709885807982, 156.8879383438245982789, 160.0646368723964968826, 163.2399473696113414057, 166.4139515028334986407, 169.5867245869675124538, 172.7583361965217071885, 175.9288507075873558181, 179.098327779000083716, 182.2668227805650139461, 185.4343871750722914559, 188.6010688598630097689, 191.7669124728935024405, 194.931959667561307812, 198.0962493599769073833, 201.2598179518737295726, 204.4226995319303405283, 207.5849260579212835438, 210.7465275218068694923, 213.9075320996092667378, 217.0679662876957848103, 220.2278550268947191126, 223.3872218156998776317, 226.546088813673058697, 229.7044769360260307364, 232.8624059402522372147, 236.0198945055811974839, 239.1769603059434383039, 242.3336200770591095102, 245.4898896781977972449, 248.6457841490992441277, 251.8013177624936876205, 254.9565040726154531201, 258.1113559600635292476, 261.2658856733274573022, 264.4201048672654226032, 267.5740246387934564975, 270.7276555600197234252, 273.8810077090356125534, 277.0340906985554608115, 280.1869137025789273745, 283.3394854812340776893, 286.4918144039449073539, 289.6439084710541581322, 292.7957753340206884025, 295.9474223143002163763, 299.0988564210088315479, 302.2500843674601579743, 305.4011125866593550246, 308.5519472458301713529, 311.7025942600449501037, 314.8530593050217504607, 318.0033478291475428826, 321.1534650647816996812, 324.3034160388896917843, 327.4532055830529745348, 330.6028383428974627138, 333.7523187869797241344, 336.9016512151670322021, 340.0508397665446839635, 343.1998884268814873563, 346.3488010356820280738, 349.4975812928522233178},
{30.24499100461545212042, 35.00401580472269102984, 39.18857461880182292162, 43.09932979894600630104, 46.84575769217668291654, 50.4817989666332570086, 54.0384447813603834862, 57.5352772423348511217, 60.9855035761415580592, 64.398467184761726264, 67.7810243792848912795, 71.1383544687693692248, 74.4744633692373037165, 77.7925107930894395891, 81.0950306522301973672, 84.3840840880083965999, 87.6613684884184409123, 90.928296887208930402, 94.1860569146788575117, 97.4356553121054704542, 100.6779520518438802585, 103.9136868417170650316, 107.1434999618814469634, 110.3679488244581798876, 113.5875212640020284991, 116.8026463003486526948, 120.0137029265184676872, 123.2210273385592115553, 126.4249189252556886383, 129.625645262637661568, 132.8234463037541876264, 136.0185379131210761563, 139.2111148639880098174, 142.4013533925575868934, 145.5894133846844285851, 148.7754402560556541519, 151.9595665754267841386, 155.141913471435543238, 158.3225918562992484928, 161.5017034939120294907, 164.6793419351866467856, 167.8555933396954768798, 171.0305371995739875924, 174.2042469791163373856, 177.3767906814062451162, 180.5482313516003547752, 183.7186275250475541112, 186.8880336272317874002, 190.0565003315244920593, 193.2240748798911305312, 196.3908013709863787413, 199.5567210194717043872, 202.7218723898788974661, 205.8862916079085623897, 209.0500125516813209813, 212.2130670251413698658, 215.375484915538713232, 218.5372943366809167913, 221.6985217594418346642, 224.8591921308386382792, 228.0193289828356008604, 231.1789545319000852441, 234.3380897702202037231, 237.4967545493922666391, 240.6549676572973803602, 243.8127468888086712972, 246.9701091109021340744, 250.127070322683777869, 253.2836457107925056917, 256.4398497005910903422, 259.5956960035159206414, 262.7511976609192045608, 265.9063670847044474274, 269.0612160950267680895, 272.2157559553035360732, 275.3699974047575284538, 278.5239506886939861185, 281.6776255866943075882, 284.8310314388924035144, 287.9841771704847268945, 291.1370713146115007457, 294.2897220337345178864, 297.4421371396259384706, 300.5943241120726297112, 303.7462901163916639558, 306.8980420198445145877, 310.0495864070301745381, 313.200929594330790194, 316.3520776434773839106, 319.5030363742977687731, 322.6538113767037842842, 325.8044080219704519128, 328.9548314733555209835, 332.1050866961041089791, 335.2551784668797009357, 338.4051113826596289072, 341.5548898691302764011, 344.7045181886146191037, 347.8540004475622995468, 351.0033406036302193636},
{31.31266698432240771286, 36.11647226741219061436, 40.33419255740740675334, 44.27181642770980684664, 48.040942469232712459, 51.6966424791159476984, 55.2706047350981868879, 58.7828806163147200358, 62.2470125825598575834, 65.6725939675958795015, 69.0666735134551825685, 72.4345824578990185124, 75.780449111620829564, 79.1075334348657762905, 82.418452592923251093, 85.7153376855374448722, 88.9999454861751467741, 92.2737398863156658798, 95.5379524108319996128, 98.7936279483148402261, 102.0416598292473047363, 105.2828170946252354668, 108.5177659491295206518, 111.7470868226747928309, 114.9712880732760781055, 118.1908170914767702461, 121.4060693732594420818, 124.6173959892942350653, 127.825109776995401107, 131.0294905070312510342, 134.2307892200803312226, 137.4292318874976096343, 140.6250225174667661283, 143.8183458035532211786, 147.0093693934582125971, 150.1982458408421209635, 153.3851142913338239521, 156.570101944530247635, 159.7533253263620243214, 162.9348914002394029235, 166.1148985405798848529, 169.293437388412578052, 172.4705916055667824361, 175.6464385413386182769, 178.8210498233762073804, 181.9944918827419686256, 185.1668264216296578488, 188.3381108309780072434, 191.5083985641876086806, 194.6777394722772645273, 197.8461801050815635565, 201.0137639824695814709, 204.1805318390363702437, 207.3465218452687857557, 210.5117698078025005461, 213.6763093510572998688, 216.8401720822543269973, 220.0033877415746685917, 223.1659843390075988798, 226.3279882792539645032, 229.4894244758914320023, 232.6503164558701461203, 235.8106864552868200078, 238.9705555072799160303, 242.1299435227962767327, 245.2888693648985455605, 248.44735091721145778, 251.6054051470422881399, 254.7630481636553051482, 257.9202952721310525472, 261.0771610231978405338, 264.2336592593842797829, 267.3898031578074239913, 270.5456052698805771614, 273.7010775581976159051, 276.8562314308263803184, 280.0110777732219564673, 283.1656269779512103952, 286.3198889724024769219, 289.4738732446386282669, 292.6275888675376478752, 295.7810445213521384675, 298.9342485148077462289, 302.0872088048501496053, 305.2399330151409218869, 308.3924284533941261032, 311.5447021276378454616, 314.696760761477910032, 317.8486108084347774054, 321.0002584654187966108, 324.1517096854038728856, 327.3029701893548045191, 330.4540454774592360677, 333.6049408397112238804, 336.7556613658898035073, 339.9062119549726515049, 343.0565973240219171125, 346.206822016576536009, 349.3568904105828052984, 352.5068067258926748529},
{32.3787123272001409191, 37.22636617156300076203, 41.47664797477027456581, 45.44071723301148932365, 49.2322313043914457752, 52.90735890227561556358, 56.4984643329480392278, 60.0260541702641967029, 63.5039962022376396411, 66.9421264616283529721, 70.3476806917745726082, 73.7261378951131801927, 77.0817455288926110457, 80.4178612322973420668, 83.7371833689049331843, 87.0419113369253336351, 90.3338599435622093429, 93.6145428295412539553, 96.8852344998126943731, 100.1470172318879773702, 103.4008170826076845768, 106.6474318978057354391, 109.8875533633819035279, 113.1217845540330673181, 116.3506540366328419571, 119.5746273065886680958, 122.7941161378728237521, 126.009486285195401136, 129.2210638730537214947, 132.4291407298004744979, 135.6339788676769536062, 138.8358142665967451816, 142.0348600865773739975, 145.2313094084300904602, 148.4253375827101653553, 151.6171042516058766016, 154.8067550963792551868, 157.9944233534063955661, 161.1802311342321311368, 164.3642905789252183634, 167.5467048670707138538, 170.7275691067170568584, 173.9069711183145948094, 177.0849921279909918328, 180.2617073822908367116, 183.4371866946704421216, 186.6114949325121004243, 189.7846924521475731257, 192.9568354883125358351, 196.1279765035553187298, 199.2981645023649349287, 202.4674453141410697482, 205.635861848582884944, 208.8034543266082233419, 211.9702604895170004691, 215.1363157887714734319, 218.3016535584727872434, 221.4663051723603392114, 224.6303001869419366956, 227.7936664721733310518, 230.9564303309412028703, 234.1186166084604495477, 237.280248792571652724, 240.4413491058153182441, 243.6019385900637152561, 246.7620371844070512312, 249.9216637969167373491, 253.0808363708432849744, 256.2395719457487834249, 259.3978867140229596887, 262.5557960731866680369, 265.7133146743465745196, 268.8704564671291605568, 272.0272347413904284369, 275.1836621659693791071, 278.3397508247280406893, 281.4955122500981993322, 284.6509574543347125748, 287.806096958657098926, 290.9609408204447593111, 294.1154986586364891827, 297.2697796774717025485, 300.4237926886988510948, 303.57754613236574223, 306.7310480962967151536, 309.8843063343528143932, 313.0373282835631087491, 316.1901210802080544213, 319.3426915749292181767, 322.4950463469336921577, 325.6471917173560862556, 328.7991337618360233693, 331.9508783223645396467, 335.1024310184486633995, 338.2537972576396747443, 341.4049822454670991178, 344.5559909948173311572, 347.7068283347928936846, 350.8574989190856851403, 354.0080072338951357243},
{33.44322284224617902737, 38.33384112532052318733, 42.61611083900191590049, 46.606217617015975691, 50.41981842379419044861, 54.11414697590573815154, 57.7222239038324136793, 61.2649977942285952248, 64.7566524653716265652, 68.2072598402843005464, 71.6242375368343651811, 75.013208377903500216, 78.3785358805325999407, 81.7236729133261755621, 85.0513970718083000421, 88.3639744588304703194, 91.6632766127773631961, 94.9508658515602464982, 98.2280587724546048172, 101.4959743028627688035, 104.7555706079235316573, 108.0076738193483191235, 111.253000666992802384, 114.4921765008095056915, 117.7257497834539341273, 120.9542038493838161786, 124.1779665244855729313, 127.3974180549734301549, 130.612897688299066426, 133.8247091705074032525, 137.033125365977885442, 140.2383921613306992183, 143.4407317816149428881, 146.6403456210025398145, 149.8374166701260325141, 153.0321116064945245247, 156.2245826020530713955, 159.4149688921407905299, 162.6033981422715482888, 165.7899876428706586372, 168.9748453570189475596, 172.1580708421270743983, 175.3397560630917183284, 178.5199861117186389949, 181.698839844916571024, 184.8763904522767834606, 188.0527059620819222374, 191.2278496934756448856, 194.4018806614245720999, 197.5748539401785068396, 200.7468209901532902582, 203.9178299524984302911, 207.0879259150486999428, 210.2571511528788733647, 213.4255453462702080446, 216.5931457785451232982, 219.7599875159236219476, 222.9261035712937675687, 226.0915250535626510882, 229.2562813040584872535, 232.4204000212843593976, 235.5839073751759718479, 238.7468281118864490928, 241.9091856500080998642, 245.0710021690419063379, 248.2322986908384041362, 251.393095154656970057, 254.5534104864229504004, 257.713262662702361592, 260.8726687698610627824, 264.0316450588284657875, 267.1902069958442602987, 270.3483693095296435134, 273.5061460345915923993, 276.6635505524393175353, 279.8205956289657676285, 282.977293449723545919, 286.1336556527035334176, 289.2896933589056095023, 292.445417200873873057, 295.6008373493534825678, 298.7559635382124631066, 301.9108050877594063899, 305.065370926576771303, 308.2196696119793478877, 311.3737093491982642172, 314.5274980093825924062, 317.6810431465030582371, 320.8343520132354996143, 323.9874315758954821818, 327.1402885284898036208, 330.2929293059454458503, 333.4453600965718170892, 336.5975868538078193037, 339.7496153073013414505, 342.9014509733651797861, 346.0530991648500916676, 349.204565000472670314, 352.3558534136329594164, 355.5069691607541853457},
{34.50628559554847640811, 39.43902818145213018397, 43.75273681975777861908, 47.76848793636236385408, 51.60388282088384199503, 55.31719035349867253035, 58.942069039056836484, 62.4998971106407229919, 66.0051656740843084053, 69.468176124146152455, 72.8965231069550014482, 76.2959695266144884503, 79.6709920248431159573, 83.0251363632893015302, 86.3612574879766420349, 89.6816866764655344374, 92.9883509438706348033, 96.2828602493614613521, 99.5665724217770368749, 102.8406423193479587937, 106.106059612001303863, 109.3636782087221139901, 112.6142394531651342741, 115.8583906055964949531, 119.0966997140068006538, 122.329667687208530394, 125.5577381768806101047, 128.781305727268467591, 132.0007225430367214759, 135.2163041458122214107, 138.4283341301988371491, 141.637068184917531633, 144.8427375103121887518, 148.0455517369812215072, 151.2457014297461948695, 154.4433602450967619012, 157.6386867975876591132, 160.8318262806161249912, 164.0229118789842645356, 167.2120660042035312963, 170.3994013782877357468, 173.5850219875464537468, 176.7690239244315607367, 179.9514961326498952437, 183.1325210684128830181, 186.3121752887535178774, 189.4905299762266440724, 192.6676514079597990551, 195.8436013758908371654, 199.0184375640765254191, 202.1922138881521090594, 205.3649808013402230808, 208.5367855708279134987, 211.7076725278361225223, 214.8776832942829946654, 218.0468569885794174915, 221.2152304127829309916, 224.3828382230667334758, 227.549713085227494818, 230.7158858167536445432, 233.8813855168001958681, 237.0462396852631852378, 240.2104743320132374767, 243.3741140772308942498, 246.5371822436838672587, 249.6997009416963424647, 252.8616911474812023277, 256.0231727754361274728, 259.1841647449427700555, 262.3446850421535139934, 265.5047507772018540943, 268.6643782372293629867, 271.8235829355839032924, 274.9823796575096042749, 278.1407825026186557314, 281.2988049244077414627, 284.4564597670575610097, 287.613759299732040765, 290.7707152485742251661, 293.9273388265782126416, 297.0836407615006376982, 300.2396313219609054486, 303.3953203418664868662, 306.5507172432879314077, 309.7058310578977154166, 312.8606704470775024595, 316.0152437207897415322, 319.1695588553016788439, 322.3236235098427269578, 325.4774450422696488643, 328.6310305238081095423, 331.7843867529337662118, 334.9375202684511595247, 338.0904373618241855752, 341.2431440888078319105, 344.3956462804271130236, 347.5479495533457092897, 350.7000593196636684963, 353.8519807961806445548, 357.0037190131584998865},
{35.56797997407609557285, 40.54204730542680501752, 44.88666890065881137216, 48.92768514199371895363, 52.78458986358995853371, 56.51665914515783554984, 60.1581720549899550224, 63.7309248498730103414, 67.2497076893041127861, 70.7250453807770010013, 74.1647050123903518134, 77.5745860214585636234, 80.9592753816762667203, 84.3224095184060164646, 87.6669189277022121475, 90.9951985934016500725, 94.3092297996807410984, 97.6106691466003261297, 100.9009148607398778367, 104.1811570323701793959, 107.4524162477954518951, 110.7155736957347710847, 113.9713949113965926837, 117.2205487058626783966, 120.4636224055245024306, 123.7011342308557762757, 126.9335434340297970578, 130.1612586637580191467, 133.3846449153771154316, 136.6040293426446953042, 139.8197061467251354202, 143.0319407117801024618, 146.2409731214370864348, 149.4470211633591251582, 152.6502829081407616852, 155.850938932326205884, 159.0491542423962458652, 162.2450799462927124216, 165.4388547108386666524, 168.6306060368127733826, 171.8204513781006019704, 175.0084991270080372571, 178.1948494842774759694, 181.3795952294367073891, 184.5628224047087958759, 187.7446109237209665672, 190.925035114593995783, 194.1041642056093410381, 197.2820627604899904171, 200.4587910693532315817, 203.6344055005673309969, 206.8089588180436095121, 209.9825004678995434176, 213.1550768379200952566, 216.3267314928093562346, 219.4975053878511205587, 222.6674370632755849872, 225.8365628213519860224, 229.0049168879870012645, 232.1725315604005943539, 235.3394373422700188148, 238.5056630675750016839, 241.6712360142394043067, 244.8361820085441165129, 248.0005255211802177851, 251.1642897557185260938, 254.3274967301898400724, 257.4901673523980010798, 260.6523214895241066235, 263.8139780325237212928, 266.9751549557688317715, 270.1358693723417821751, 273.2961375853488168764, 276.4559751355855546248, 279.6153968458552036159, 282.7744168622121534007, 285.933048692378355366, 289.0913052415572881422, 292.2491988458499998874, 295.4067413034594652904, 298.563943903853062782, 301.7208174550381661366, 304.8773723090924778353, 308.0336183860786537056, 311.1895651964618422254, 314.3452218621388662049, 317.5005971361788020305, 320.6556994213665671729, 323.8105367876337258367, 326.9651169884539904728, 330.1194474762747666607, 333.2735354170505011639, 336.4273877039384947228, 339.5810109702131850339, 342.734411601450649008, 345.8875957470311788993, 349.0405693310042202639, 352.1933380623566904444, 355.3459074447226970084, 358.4982827855699216427},
{36.62837858971343507619, 41.643008631132493502, 46.01803876636435956249, 50.08395419714843394157, 53.96209269075680683649, 57.7127112641518718185, 61.3706932749029995829, 64.9582420604165640633, 68.49043906707716198, 71.9780267878152583915, 75.4289404075530519766, 78.8492125271880596232, 82.2435377933728849487, 85.6156411670123638062, 88.9685269708056110485, 92.3046524854459831616, 95.6260521017978565403, 98.9344280952278867682, 102.2312182828947403429, 105.5176473086718902083, 108.7947661022490410288, 112.0634826460906670164, 115.3245862532202979787, 118.5787669321435714782, 121.8266309849298915936, 125.0687136837361195994, 128.3054896574677332104, 131.5373814663452037838, 134.7647667297176553252, 137.9879840893374217586, 141.2073382281433025562, 144.4231041176215611147, 147.6355306309653022691, 150.8448436316494927772, 154.0512486256043664326, 157.2549330483940323875, 160.4560682455800140773, 163.6548111939479333694, 166.8513060028834510149, 170.0456852284358300147, 173.2380710271503323183, 176.4285761723129638226, 179.6173049516235921245, 182.8043539623335943974, 185.9898128174248319347, 189.1737647743679074694, 192.3562872963002469547, 195.5374525540456509142, 198.7173278762062882573, 201.895976153555250537, 205.0734562031101296936, 208.2498230965491792858, 211.4251284570199297202, 214.5994207278680394017, 217.7727454163672164758, 220.9451453151473236645, 224.1166607036874254713, 227.2873295319553718113, 230.457187588028721532, 233.6262686513177003553, 236.7946046328246887804, 239.9622257037124347811, 243.1291604133114011717, 246.2954357975725282152, 249.4610774788627934319, 252.6261097579052212029, 255.7905556985806777734, 258.9544372062343816244, 262.1177751000642794374, 265.2805891801101805839, 268.4428982893108589401, 271.6047203710504013782, 274.7660725225742003706, 277.9269710446185414928, 281.0874314875651941167, 284.2474686944033135909, 287.4070968407549036999, 290.5663294721967184623, 293.7251795390904958553, 296.8836594291145444838, 300.0417809976727122268, 303.1995555963414467702, 306.3569940995018298143, 309.5141069292909693597, 312.6709040789958264113, 315.8273951350023086785, 318.9835892974031737733, 322.1394953993598499343, 325.2951219253056164018, 328.4504770280706108783, 331.6055685450027791751, 334.7604040131530908329, 337.9149906835880594633, 341.0693355348877789032, 344.2234452858832722689, 347.377326407682911477, 350.5309851350339646738, 353.6844274770619367593, 356.8376592274272555157, 359.9906859739359972835},
{37.68754805180205317425, 42.74201354048884837766, 47.14696800152473906661, 51.23742930903326104237, 55.13653343056308703409, 58.90549360675247638589, 62.57978215651488453208, 66.1819991773134881905, 69.7275100652628767794, 73.2272695778860606038, 76.6893768756630905684, 80.1199945199303918633, 83.5239222967487062328, 86.9049716699102358212, 90.2662191386342591037, 93.6101829275400314985, 96.9389494155071955746, 100.254265621479014428, 103.5576081722484323779, 106.8502356071019211492, 110.1332286416824072508, 113.4075215780713158846, 116.6739271022999154522, 119.9331560735123325686, 123.1858334714907261817, 126.432511363383805554, 129.6736795331835548097, 132.9097742608491188006, 136.1411856235321289814, 139.3682636067113806942, 142.5913232497249559528, 145.8106490023227260088, 149.0264984323263875498, 152.2391053963430532829, 155.4486827636197114002, 158.6554247660127767045, 161.8595090335499362172, 165.0610983643424275141, 168.2603422690373400963, 171.4573783231078325994, 174.6523333547039019339, 177.8453244912512439879, 181.0364600842775259365, 184.225840528898297435, 187.4135589918791765676, 190.599702060104873261, 193.7843503195483643068, 196.9675788733809146385, 200.1494578066443140934, 203.3300526038794137679, 206.5094245252364925123, 209.6876309458561468525, 212.8647256626822711819, 216.0407591723332805077, 219.2157789231992450901, 222.3898295445388715725, 225.5629530550111991304, 228.7351890527841134337, 231.9065748891083447164, 235.0771458270256828582, 238.2469351866888276252, 241.4159744786034886842, 244.5842935259575936115, 247.7519205770748231446, 250.9188824089176867935, 254.0852044224668660474, 257.2509107307167858911, 260.416024239950789222, 263.5805667248915647616, 266.7445588982624853509, 269.9080204752422761525, 273.0709702332481113363, 276.233426067440107457, 279.3954050423026133668, 282.5569234396241447535, 285.7179968031678020898, 288.8786399802971309405, 292.0388671607982728205, 295.198691913117598107, 298.3581272182145340339, 301.5171855012117585727, 304.6758786610091124479, 307.8342180980132995836, 310.9922147401225360499, 314.1498790670936235453, 317.3072211334083369596, 320.4642505897464127163, 323.6209767031637044237, 326.7774083760661452352, 329.9335541640629425428, 333.0894222927758592824, 336.2450206736754439936, 339.400356919009602445, 342.5554383558849066984, 345.710272039556467861, 348.8648647659780160745, 352.0192230836599993171, 355.173353304879998837, 358.3272615162865342992, 361.4809535889343697273},
{38.74554963074332533582, 43.83915559613160980124, 48.2735691317974348699, 52.38823500317661473721, 56.30804426784111671476, 60.09514308985669809738, 63.7855782873083538386, 67.4023369685844188592, 70.9610615382079456564, 74.4729138809759461298, 77.9461532197193924454, 81.3870690285311394938, 84.8005638170946274896, 88.1905336095803468424, 91.5601255011549626468, 94.9119173614065318082, 98.2480464806064367913, 101.5703037223765787604, 104.8802037678458109393, 108.1790384135318155839, 111.4679176191556272438, 114.7478015443145557875, 118.0195258524575963273, 121.2838219135555630373, 124.5413330923163751842, 127.7926279979288899624, 131.0382113504122743362, 134.2785329593561924759, 137.5139951944274573197, 140.7449592408952947766, 143.9717503689846575104, 147.1946623971394121793, 150.4139614920726746282, 153.6298894198172413286, 156.8426663397176677286, 160.0524932158644140913, 163.2595539067108311629, 166.46401698268301441, 169.6660373128523142298, 172.8657574547085028721, 176.0633088753813996409, 179.2588130280288517162, 182.452382303322158142, 185.6441208728474775235, 188.8341254386714109339, 192.0224859011868749433, 195.2092859555793265834, 198.3946036257679249141, 201.5785117434289646992, 204.7610783786578192727, 207.9423672279367006612, 211.1224379643212185119, 214.3013465541165354161, 217.479145543765500063, 220.6558843202013990231, 223.8316093475144594495, 227.0063643824336616646, 230.1801906708252317397, 233.3531271271492551395, 236.5252104985902275176, 239.6964755153810435596, 242.8669550286687186943, 246.0366801371204980365, 249.2056803033379345183, 252.3739834610314711646, 255.5416161138068784136, 258.7086034263257324983, 261.8749693085233992195, 265.0407364934983587116, 268.2059266096250114874, 271.370560247387346134, 274.5346570213821636957, 277.6982356278971989203, 280.8613138984308096805, 284.0239088494853636707, 287.1860367289355495574, 290.3477130592451544033, 293.5089526777810105965, 296.6697697744505003436, 299.8301779268689308633, 302.9901901332450102833, 306.1498188431563445873, 309.3090759863721476205, 312.4679729998670397493, 315.6265208531577567407, 318.7847300720836665381, 321.9426107611420807961, 325.1001726244803464699, 328.2574249856385182626, 331.4143768061289633472, 334.5710367029324625432, 337.7274129649841819872, 340.8835135687172382411, 344.0393461927264158504, 347.194918231609873176, 350.3502368090423483065, 353.5053087901294147103, 356.6601407930887025503, 359.8147392003006661505, 362.969110168768413918},
{39.80243983079326852703, 44.93452135081094554623, 49.39794653155941261957, 53.53648706428156819261, 57.47674838253129660885, 61.28178756675501009454, 64.98821226605180582508, 68.6193873762271263956, 72.1912257342410832757, 75.7150914775287567399, 79.1994001716219444267, 82.6505653009515099693, 86.0735897935960043376, 89.472452376650159663, 92.8503692266328711398, 96.2099766106413286243, 99.5534616940887029017, 102.8826583181194425358, 106.1991184878872704253, 109.504166638623382358, 112.7989414476956029701, 116.0844284821973001531, 119.3614859967962849191, 122.6308655397100797168, 125.8932285732796220448, 129.1491599998757378899, 132.399179259448591821, 135.6437495031608764279, 138.8832852292217416827, 142.1181586794762014493, 145.3487052297684232502, 148.5752279575297920485, 151.7980015321868460221, 155.0172755448103943758, 158.2332773707538783495, 161.4462146412684858212, 164.6562773860671490864, 167.8636398976726841725, 171.0684623594778899411, 174.2708922722772078469, 177.4710657082273943192, 180.6691084164724054254, 183.8651368008043240425, 187.0592587865558418502, 190.2515745912961027861, 193.4421774117248050063, 196.6311540373455820122, 199.8185853999821742548, 203.0045470669264262551, 206.1891096844328222396, 209.3723393773654694825, 212.5542981100320490059, 215.7350440125823756696, 218.9146316767881033264, 222.0931124245393795435, 225.2705345519812018396, 228.4469435518563448175, 231.6223823163142866063, 234.7968913221792894739, 237.9705088004396012756, 241.1432708915185403245, 244.3152117877127147723, 247.4863638640291877554, 250.6567577985189677187, 253.8264226830861801994, 256.9953861256484518859, 260.1636743444325243307, 263.3313122551083012553, 266.4983235513930384905, 269.6647307796940226657, 272.8305554083018323344, 275.9958178915962520253, 279.1605377296823538634, 282.3247335238345128602, 285.4884230280906077831, 288.6516231973068813881, 291.8143502319554575317, 294.9766196199209592419, 298.1384461755297097769, 301.2998440760243376557, 304.4608268956779918466, 307.6214076377255807908, 310.7815987642742816093, 313.9414122243418498567, 317.1008594801588420863, 320.2599515318596075359, 323.4186989406766912325, 326.5771118507440121526, 329.7352000096057420734, 332.8929727875201292417, 336.0504391956405109833, 339.2076079031493740582, 342.3644872534154912533, 345.5210852792388340767, 348.6774097172430867472, 351.8334680214711232054, 354.9892673762347182617, 358.1448147082660118034, 361.3001166982148002536, 364.4551797915325643578},
{40.85827088672840192539, 46.02819105278710054563, 50.52019721854089739402, 54.68229336325653641987, 58.64276077774657519628, 62.46554663805285762038, 66.18780648600253946077, 69.833274265767379026, 73.418127008563688268, 76.95392647354301818, 80.4492410295481319546, 83.9106054047638928558, 87.3431207442651425767, 90.7508467008633559983, 94.1370670803885958138, 97.5044753490706266538, 100.8553075499147760268, 104.1914396650475878213, 107.5144603176026717814, 110.8257259822498620302, 114.1264035428168299482, 117.4175035359182893687, 120.699906430716154398, 123.9743836284955273722, 127.2416144076616353518, 130.5021997192740167703, 133.7566735103793739926, 137.0055120880214505122, 140.2491419166162642595, 143.4879461524161542826, 146.7222701521837399044, 149.9524261428107373739, 153.178697200126075333, 156.4013406554674607093, 159.6205910255256832871, 162.8366625428991187787, 166.0497513505312968965, 169.2600374118666546822, 172.4676861794892122646, 175.6728500577077267548, 178.8756696886396194084, 182.0762750865337052631, 185.2747866411337541712, 188.4713160076464236632, 191.6659668982013748732, 194.8588357874707548208, 198.0500125432644640644, 201.2395809913688747715, 204.4276194225956261572, 207.6142010489101423067, 210.7993944145813178474, 213.983263767505772618, 217.1658693951888540351, 220.347267929291068137, 223.5275126221571551996, 226.7066535983226501364, 229.8847380836287666574, 233.0618106142619158752, 236.2379132277616947432, 239.4130856378045522528, 242.5873653943643561636, 245.7607880306713600707, 248.9333871982339134333, 252.1051947910495376376, 255.2762410600110547552, 258.4465547184070439032, 261.6161630393220865844, 264.7850919456593998329, 267.9533660934351325083, 271.1210089489286027245, 274.2880428602150387688, 277.4544891235560507697, 280.6203680450773265299, 283.7856989981222371216, 286.9505004766335698016, 290.114790144882967593, 293.2785848838384020524, 296.4419008344337502623, 299.604753437980949561, 302.7671574739439665682, 305.9291270952746797067, 309.0906758614935072362, 312.2518167696820137904, 315.4125622835406194203, 318.5729243606517588613, 321.7329144780772559994, 324.8925436564081660673, 328.0518224823757866076, 331.2107611301238505972, 334.3693693812340049492, 337.5276566435894679817, 340.6856319691551817642, 343.8433040707467682775, 347.0006813378551073022, 350.1577718515883298837, 353.3145833987884201962, 356.4711234853754012532, 359.6273993489682108227, 362.7834179708278212745, 365.9391860871648923157},
{41.91309119634458476528, 47.12023926304825270298, 51.64041155208319947059, 55.82575458676858204399, 59.80618901287532732032, 63.6465323720209084267, 67.38447583284268935617, 71.0441140962059381173, 74.6418824622344458656, 78.1895359073059705514, 81.695792232246508472, 85.167304769528127091, 88.6092707773766378255, 92.0258291328293124903, 95.4203298782757565233, 98.7955225274476898183, 102.153691040441568093, 105.496752733299073615, 108.8263321645928802659, 112.1438172679217350516, 115.4504026373652713667, 118.7471233530228506928, 122.0348817313100916805, 125.3144687088975631492, 128.5865811045685155594, 131.8518356781454545749, 135.1107806744317514888, 138.363905373278299382, 141.6116480448740953284, 144.8544026190262604187, 148.0925243095526697564, 151.3263343837256497627, 154.5561242275956022481, 157.7821588278688705936, 161.0046797675673642008, 164.2239078143220543775, 167.4400451656448203555, 170.6532774039894776124, 173.8637752051835278392, 177.0716958363813119609, 180.2771844736715687513, 183.4803753645723391301, 186.6813928566353583087, 189.8803523100829411144, 193.0773609096738821343, 196.2725183887316080502, 199.4659176763810592689, 202.6576454774615310043, 205.84778279325573953, 209.0364053900562764084, 212.2235842216434766099, 215.4093858109444183227, 218.5938725954575393844, 221.7771032404417529056, 224.959132923366992125, 228.1400135926916280747, 231.3197942036602671083, 234.4985209334939784292, 237.6762373780664623462, 240.8529847319177228497, 244.0288019532461460144, 247.2037259153360421261, 250.3777915457169142759, 253.5510319542097783962, 256.7234785508920709412, 259.8951611549037358899, 263.0661080949210161987, 266.2363463020396046311, 269.4059013957336934182, 272.5747977634908627335, 275.743058634663597741, 278.9107061490256023072, 282.0777614204741881091, 285.2442445962781707345, 288.4101749122333002348, 291.5755707440537710937, 294.740449655298341277, 297.9048284421026450707, 301.0687231749650626411, 304.23214923781170642, 307.3951213645464334959, 310.5576536732740593668, 313.7197596983689249272, 316.8814524205464730389, 320.0427442950823623898, 323.2036472783117420569, 326.364172852530504016, 329.5243320494105107703, 332.6841354720318618029, 335.8435933156271271729, 339.002715387125060437, 342.1615111235745358588, 345.3199896095232737761, 348.4781595934202668937, 351.6360295031056488848, 354.7936074604470099967, 357.9509012951768219332, 361.107918557981649849, 364.2646665328901701803, 367.421152249003649887},
{42.9669456985971009999, 48.21073539741246349877, 52.75867384888918110164, 56.96696488296545520639, 60.96713385466949154042, 64.82484994641067738603, 68.5783283084022838744, 72.25201652043633762532, 75.8626025163847123209, 79.4220302960249445451, 82.9391638777025354515, 86.4207726777685842255, 89.8721480554609822807, 93.2975064820090998191, 96.7002628997974192466, 100.0832217629249579119, 103.4487140235106254602, 106.7986975527756743964, 110.1348321849093926823, 113.4585367501775301194, 116.7710330713679034145, 120.0733803578032424576, 123.3665024153484639465, 126.6512094059113314574, 129.9282154189489464476, 133.1981527878338020545, 136.4615838494583111482, 139.7190106762245554771, 142.9708831857764511423, 146.21760594216190652, 149.4595438934548486692, 152.6970272389027036036, 155.9303555789509949425, 159.1598014708695155012, 162.385613488884953223, 165.6080188690531704324, 168.8272258043595072639, 172.0434254438106494971, 175.2567936398972243131, 178.4674924812488735195, 181.6756716411820147249, 184.8814695678547490795, 188.0850145376615221264, 191.2864255901418078768, 194.4858133599010567459, 197.683280818737219624, 200.8789239392442549058, 204.0728322895550029217, 207.2650895675335285669, 210.4557740815863243602, 213.6449591842961305946, 216.8327136642619318273, 220.0191021008297662069, 223.2041851858015522349, 226.3880200156969342456, 229.570660357702754736, 232.7521568920650662568, 235.9325574333503628285, 239.1119071327182399465, 242.290248663100540791, 245.4676223889668077545, 248.6440665221679819189, 251.8196172651859367559, 254.994308942972343593, 258.1681741244337842916, 261.3412437345086011802, 264.513547157682700403, 267.6851123337046872888, 270.8559658461838404525, 274.0261330046862612342, 277.1956379208839781111, 280.3645035792579021985, 283.5327519028075070093, 286.7004038141772386992, 289.8674792925713401449, 293.0339974267944608931, 296.1999764647246610488, 299.3654338594977950612, 302.5303863126574263654, 305.6948498145020649996, 308.858839681841364661, 312.0223705933547231344, 315.1854566227292891915, 318.3481112697395034022, 321.5103474894168250418, 324.6721777194460766177, 327.8336139059137423066, 330.9946675275234721447, 334.1553496183848683195, 337.3156707894732727495, 340.4756412488506555805, 343.6352708207307502747, 346.7945689634652282952, 349.9535447865218974951, 353.1122070665205917124, 356.2705642623875486005, 359.4286245296846069747, 362.5863957341654567247, 365.7438854646074101662, 368.9011010449637034677},
{44.01987620547096356426, 49.2997442043539280121, 53.87506292784085779202, 58.10601243481444555528, 62.12568985723225325489, 66.00059822192881776969, 69.7694655895711846718, 73.4570849247367790262, 77.0803914294972706203, 80.6515141294635667297, 84.1794601926125461548, 87.6711127103735129832, 91.1318552171187456465, 94.5659802156942447049, 97.9769662651611128539, 101.3676716951902662644, 104.7404735587158446053, 108.0973695306044174787, 111.4400540827611341489, 114.7699763975609390226, 118.0883850592686645027, 121.3963630037390721594, 124.6948551778231580763, 127.9846906660401476028, 131.2666005648501482298, 134.5412325507729762979, 137.8091628509259951553, 141.070906152979743034, 144.3269238660025445837, 147.5776310506899486289, 150.8234022678250813271, 154.0645765410901130031, 157.3014615900447826752, 160.5343374579990272976, 163.7634596353230984538, 166.9890617597775039191, 170.2113579604686361579, 173.4305449001244106445, 176.6468035608484087743, 179.8603008108300144989, 183.0711907832650660676, 186.279616093672107068, 189.485708917638074009, 192.6895919476111013443, 195.8913792445337014023, 199.0911769977639695409, 202.2890842047762261236, 205.485193280494394356, 208.6795906047343904079, 211.8723570150699808949, 215.0635682514528484242, 218.2532953580818779136, 221.4416050473043710663, 224.6285600297228988669, 227.8142193141592751232, 230.9986384806780303493, 234.1818699294844806291, 237.3639631081776245855, 240.544964719547826962, 243.7249189118570027772, 246.9038674533192980987, 250.0818498923084388971, 253.2589037046500760764, 256.4350644292102801602, 259.6103657928620208466, 262.784839825797610658, 265.9585169680546550097, 269.1314261680342852259, 272.303594973711857292, 275.4750496171705889774, 278.6458150930266713807, 281.8159152312592696385, 284.9853727649096924041, 288.1542093930701419777, 291.3224458395432323086, 294.4901019075183370896, 297.6571965305793310254, 300.8237478203300001711, 303.9897731108979598105, 307.1552890005550131781, 310.3203113906712326176, 313.4848555222014013269, 316.6489360098856026533, 319.8125668743304944081, 322.9757615721239894659, 326.1385330241235319961, 329.300893642046778978, 332.4628553534831519889, 335.6244296254353104864, 338.7856274864910222591, 341.9464595477180868179, 345.1069360223678295911, 348.2670667444661630519, 351.4268611863652465109, 354.58632847532331654, 357.7454774091752576289, 360.9043164711518953036, 364.0628538439017834694, 367.2210974227653899161, 370.3790548283480278766},
{45.07192169428919331684, 50.38732618759618466158, 54.98965259359106497861, 59.24297997070654986373, 63.28194588014616369617, 67.17387025603796435242, 70.95798353042746252046, 74.65941691471047141232, 78.2953477644868739974, 81.8780863167188178056, 85.4167799582410166608, 88.9184231514752402135, 92.3884897612419701362, 95.8313468231358707393, 99.2505352800414276866, 102.648966312685003605, 106.0290622159513970427, 109.3928597429104209617, 112.7420873864065103401, 116.0782241535105146616, 119.4025449366682215012, 122.7161560069069438943, 126.0200231128102746192, 129.3149939663555673988, 132.6018164133799542141, 135.8811532480161593851, 139.1535943896383606926, 142.4196669669936525632, 145.6798437269673299823, 148.9345500911780542438, 152.1841701129792485815, 155.4290515339726230157, 158.6695100982584062831, 161.9058332511066924533, 165.1382833241947928958, 168.3671002903116408719, 171.592504155227439871, 174.8146970423327504547, 178.0338650159673829832, 181.2501796815576848338, 184.4637995943587274189, 187.674871503446683316, 190.8835314533876222425, 194.0899057625362930832, 197.2941118940467400951, 200.4962592332912365001, 203.6964497833942218593, 206.8947787889213815708, 210.0913352963627999985, 213.2862026588666561192, 216.4794589916785394067, 219.6711775838905487751, 222.8614272713789441351, 226.0502727751888102347, 229.2377750090921602785, 232.4239913595882809995, 235.6089759412204100973, 238.79277982974148824, 241.9754512753657725394, 245.1570358980858679823, 248.337576866810629554, 251.5171150638836969343, 254.6956892363711649783, 257.8733361353566956121, 261.0500906443503776609, 264.2259858978014032584, 267.401053390602072537, 270.5753230793799821945, 273.7488234762949715226, 276.9215817359861790091, 280.0936237362512728106, 283.264974152983582716, 286.4356565298426347369, 289.605693343088739513, 292.7751060619721758206, 295.9439152050315865535, 299.1121403926239834379, 302.2798003959798162885, 305.4469131830505333452, 308.6134959613926172131, 311.7795652183109416827, 314.9451367584652079925, 318.1102257391259644457, 321.2748467032510961414, 324.4390136105395200216, 327.6027398666059822775, 330.7660383504091955499, 333.9289214400549525791, 337.0914010370862046296, 340.2534885893633024326, 343.4151951126295801895, 346.57653121085014406, 349.7375070954050382669, 352.8981326032118443112, 356.0584172138471679301, 359.2183700657313360395, 362.377999971435918609, 365.537315432169369268, 368.6963246514921083662, 371.8550355483087216589},
{46.12311856605032289735, 51.473537981052457933, 56.1025120671074968635, 60.37794522049286012187, 64.43598555259812198135, 68.34475376448384776813, 72.14397261446204860833, 75.85910475401782503993, 79.507564811399367147, 83.1018405914534756659, 86.6512168975052606801, 90.1627973572199851202, 93.6421443976518726107, 97.0936981484684607779, 100.5210607513626549683, 103.9271952519157526555, 107.3145683589789743992, 110.6852552033937720219, 114.0410177025007947513, 117.3833641772308541291, 120.7135953884546836209, 124.0328405620821895488, 127.3420859182258046432, 130.6421975085588119542, 133.9339396766933577626, 137.2179901137340252794, 140.4949522372994915395, 143.7653654461981097344, 147.0297136740525030505, 150.2884325696667467219, 153.5419155603610133403, 156.7905190002997284459, 160.0345665643934905276, 163.2743530163734556129, 166.5101474547493428466, 169.7421961208418138004, 172.9707248376557470553, 176.1959411360885185183, 179.4180361151388550665, 182.6371860748619595045, 185.8535539543976082648, 189.0672906021667475297, 192.2785359010467432936, 195.4874197678075278797, 198.6940630431729185812, 201.8985782864471406835, 205.1010704866239726867, 208.3016377002015737618, 211.5003716245011929931, 214.697358114085307345, 217.8926776468520522973, 221.0864057455170366665, 224.2786133594554344637, 227.4693672112458674129, 230.6587301117159637993, 233.8467612468235125036, 237.0335164393051511346, 240.2190483876768274594, 243.4034068848687559071, 246.5866390185154779343, 249.7687893546932357236, 252.9499001066973905535, 256.1300112902780124637, 259.3091608665986081201, 262.4873848740483273421, 265.6647175499194161663, 268.84119144285704894, 272.0168375168961623305, 275.1916852478179764181, 278.3657627124861861298, 281.5390966717581888545, 284.7117126475091868315, 287.8836349942557053477, 291.0548869658192536947, 294.2254907774298768194, 297.3954676636326371831, 300.5648379323271361544, 303.7336210152406020786, 306.9018355151084614055, 310.0694992498123396715, 313.2366292937038206913, 316.4032420163227698299, 319.5693531187013758504, 322.734977667429087119, 325.9001301266391365031, 329.0648243880642098591, 332.2290737992968779486, 335.3928911903795587715, 338.5562888988388980623, 341.7192787932704533992, 344.8818722955713558697, 348.0440804019111256355, 351.2059137025239653578, 354.3673824003995868077, 357.5284963289438860745, 360.6892649686755222171, 363.8496974630196286851, 367.0098026332554565328, 370.1695889926706775403, 373.3290647599713315559},
{47.17350087447835357753, 52.55843268249647738333, 57.21370637008936177742, 61.51098132389493855499, 65.58788769020688700871, 69.51333153688941915763, 73.32751836281689754708, 77.05623576137566055256, 80.7171309707669037257, 84.3228658802012504159, 87.8828600275090371113, 91.4043240922806154986, 94.8929073676666818437, 98.353121695630860704, 101.7886292770200020727, 105.202444072519376165, 108.5970764064394333273, 111.9746391109224409104, 115.3369269509176926383, 118.6854770663890935059, 122.0216156600105289547, 125.3464945430742765955, 128.6611200858856623049, 131.9663763993368407748, 135.2630440791896919993, 138.5518154977709979683, 141.8333073809201039877, 145.1080712297268435204, 148.3766020160778589338, 151.6393454843046813421, 154.8967043187291736176, 158.1490433819901108361, 161.3966941870363817796, 164.6399587332587602603, 167.879112812004066531, 171.1144088669245085916, 174.3460784789741498563, 177.574334533417471399, 180.7993731162448652363, 184.0213751793544895169, 187.2405080073461052284, 190.4569265134631097349, 193.6707743878687401211, 196.8821851178604511479, 200.0912828966632584587, 203.2981834349805556478, 206.5029946874261894601, 209.7058175042400082793, 212.9067462172410926134, 216.1058691677504657366, 219.3032691831794766446, 222.4990240080997332466, 225.6932066948597306848, 228.8858859581711043603, 232.0771264975364147732, 235.2669892909162375547, 238.4555318626242338358, 241.6428085280849564916, 244.8288706177821823821, 248.0137666824586647429, 251.1975426813955920362, 254.3802421553968503997, 257.561906385925302951, 260.7425745416822361401, 263.9222838137839213278, 267.1010695405683780991, 270.2789653229587523282, 273.4560031312153945561, 276.6322134038251624115, 279.8076251392023154892, 282.9822659808094509756, 286.1561622962482316505, 289.3293392508173089937, 292.5018208759880834994, 295.6736301332071118924, 298.8447889733964920618, 302.0153183924899298366, 305.1852384833119790916, 308.3545684840807645159, 311.5233268237900083128, 314.6915311647040926155, 317.8591984421799384987, 321.0263449020114409782, 324.1929861354758650779, 327.3591371122468020952, 330.5248122113248493325, 333.6900252501249704574, 336.8547895118483926978, 340.0191177712567905389, 343.1830223189572948283, 346.3465149842984633507, 349.509607156969675539, 352.6723098073893999499, 355.8346335059613657277, 358.9965884412717923238, 362.158184437295444843, 365.3194309696733403095, 368.480337181120392223, 371.6409118960171104125, 374.8011636342356374333},
{48.22310052972619587308, 53.64206015136298333346, 58.32329666913697252539, 62.64215719721278794039, 66.73772667029803185153, 70.67968181186868709436, 74.50870170364330018665, 78.25089267056454566752, 81.92413010199545525783, 85.541246637768085445, 89.1117939812144310488, 92.6430878374894112564, 96.140862737687420874, 99.6097009081061169199, 103.05332351211705666, 106.4747945104364376078, 109.8766670724606976187, 113.2610910781083333937, 116.6298935818442234245, 119.9846400632832473244, 123.3266817530036502841, 126.6571926886778738673, 129.9771990781376445392, 133.2876028181766609956, 136.5892005169894788407, 139.8826990172425548718, 143.1687281669705483586, 146.4478514050368116389, 149.7205745957813244311, 152.98735344955612595, 156.2485997924404487185, 159.5046868928175602329, 162.7559540099559259729, 166.0027102969007415299, 169.2452381644202534542, 172.4837961926952811161, 175.7186216615874328005, 178.94993275770362946, 182.1779305063658085908, 185.4028004684461747373, 188.6247142354217385543, 191.8438307506159061333, 195.0602974801811368018, 198.2742514537418066051, 201.4858201916089089481, 204.6951225329787060105, 207.9022693774412870598, 211.1073643503767819141, 214.3105044013463113788, 217.511780343343001842, 220.711277339716215666, 223.9090753446875957298, 227.1052495026145151286, 230.2998705105036955915, 233.4930049477175345287, 236.6847155763335473101, 239.8750616152012599866, 243.0640989903808743592, 246.2518805643357238094, 249.4384563459789553286, 252.6238736834381407565, 255.8081774411946935885, 258.9914101630738692545, 262.1736122224022171695, 265.3548219605096253037, 268.5350758146299914483, 271.7144084361458827663, 274.8928528000264344804, 278.0704403062225832285, 281.2472008737081490942, 284.423163027788081666, 287.5983539812353409211, 290.7727997097645026784, 293.9465250223024883171, 297.119553626474147957, 300.2919081896821906517, 303.4636103961266442129, 306.6346810000781951891, 309.8051398756920158457, 312.9750060636236873171, 316.1442978146862755611, 319.3130326307672445969, 322.4812273032054663973, 325.6488979488119027271, 328.8160600437024090517, 331.9827284550973831253, 335.148917471230508021, 338.3146408294974942275, 341.4799117429653951216, 344.6447429253536541318, 347.8091466145894506275, 350.9731345950320650323, 354.136718218453810374, 357.2999084238585134966, 360.4627157562125171595, 363.625150384157662833, 366.7872221167706570261, 369.9489404194285799433, 373.1103144288360268381, 376.2713529672654459596},
{49.27194748006372293503, 54.7244672752636844848, 59.43134058469048721109, 63.77153786340396132515, 67.8855727705657396969, 71.8438786163652791461, 75.68759930700176264267, 79.44315395756249509103, 83.1286418405976069717, 86.7570631532456241608, 90.3380993014851201898, 93.8791690715582561393, 97.3860906685219040784, 100.8635154259720563628, 104.3152224140014166225, 107.7443247112824300407, 111.153417588775454415, 114.544687342619442147, 117.9199927767554889725, 121.2809272459555903432, 124.6288666071142574911, 127.9650067754807106561, 131.2903934922057297412, 134.6059461736865344558, 137.9124772066560013484, 141.2107076980624053077, 144.5012804361000044338, 147.784770636186815836, 151.0616949120055288461, 154.3325188126255263925, 157.597663192424113948, 160.8575096242304323522, 164.1124050230458941058, 167.3626656144439147817, 170.6085803558633356489, 173.8504138986943350081, 177.0884091629949559401, 180.3227895838910924721, 183.5537610784682682596, 186.7815137737042595781, 190.006223529294086778, 193.2280532837579201999, 196.447154247746511437, 199.6636669647720586, 202.8777222565415306153, 206.0894420675334896512, 209.2989402213425077002, 212.506323099540974804, 215.7116902523152354879, 218.9151349488722878083, 222.1167446745448732057, 225.316601580614283466, 228.5147828920951790224, 231.7113612780634883149, 234.9064051885392188924, 238.0999791614460389156, 241.2921441027465828179, 244.482957542486449319, 247.6724738691623243173, 250.8607445445534836795, 254.0478183009151548904, 257.2337413222218233832, 260.4185574109643186974, 263.6023081418428143531, 266.7850330035556706284, 269.9667695297587406664, 273.1475534201591264516, 276.3274186526095106373, 279.5063975869824710894, 282.6845210615272036533, 285.8618184823426249362, 289.0383179065398553395, 292.2140461196126880256, 295.3890287074860466743, 298.5632901236689411909, 301.736853751899451442, 304.9097419646342858707, 308.0819761777040210089, 311.2535769014268323416, 314.4245637884480283492, 317.59495567854969175, 320.7647706406539456069, 323.934026012224559555, 327.1027384362545832912, 330.2709238960122552477, 333.4385977477034198813, 336.6057747511959516578, 339.7724690989400983238, 342.9386944432081054532, 346.1044639217668661763, 349.2697901820885631617, 352.4346854041962528407, 355.5991613222340117111, 358.7632292448445561417, 361.9269000744311020209, 365.0901843253745964592, 368.2530921412722835106, 371.415633311258817228, 374.5778172854667702289, 377.7396531896793700198},
{50.32006987437750602279, 55.80569820912888212923, 60.53789246903300224084, 64.89918474989517951493, 69.03149247538275794976, 73.00599207328879403515, 76.86428388914350080498, 80.63309413839287187117, 84.33074188760238711609, 87.970391829717768953, 91.5618527103417394327, 95.1126445295002887446, 98.6286676628492245998, 102.114641322468477243, 105.5744014681231066401, 109.0111094457747865676, 112.4274019100537952941, 115.8255009627933623501, 119.2072966347067124649, 122.5744097055716578791, 125.9282402689115699632, 129.2700057786448222082, 132.6007712132744602059, 135.9214732493707174904, 139.2329398240341672652, 142.5359061072044492851, 145.8310276491649342504, 149.1188912839599174793, 152.4000242342256761146, 155.6749017626865376665, 158.943953640392451115, 162.2075696448088022418, 165.4661042572796912654, 168.7198806957284564012, 171.9691943922505460167, 175.214316004683790865, 178.4554940349768348788, 181.6929571142271280755, 184.9269160038823239324, 188.1575653542310843592, 191.3850852545226895423, 194.6096426035203817042, 197.8313923247564969266, 201.0504784470199595432, 204.2670350675133261308, 207.4811872125448468779, 210.6930516084739034377, 213.9027373738283181626, 217.1103466419974202452, 220.3159751226254848087, 223.5197126087458491698, 226.7216434357738387142, 229.9218468976898176891, 233.1203976250702652352, 236.3173659290467011376, 239.5128181147746297422, 242.7068167675650664277, 245.8994210144593701475, 249.0906867637054396912, 252.2806669243126484371, 255.4694116076181507424, 258.6569683125833019822, 261.8433820963515915807, 265.0286957314350484757, 268.2129498507514433849, 271.3961830816071426167, 274.5784321696079095505, 277.7597320933803719464, 280.9401161708986191059, 284.1196161581320372863, 287.2982623406608057591, 290.4760836188433976455, 293.6531075870650397146, 296.829360607546585918, 300.0048678791489570219, 303.1796535015685891495, 306.3537405352836901685, 309.5271510575790637297, 312.6999062149484227111, 315.8720262721471226856, 319.0435306581447905834, 322.2144380092061300082, 325.3847662093090111659, 328.5545324280915868062, 331.7237531565044273682, 334.8924442403293716362, 338.0606209117137955095, 341.2282978188571794079, 344.3954890539760873632, 347.5622081796638537796, 350.7284682537523142675, 353.8942818527747319005, 357.0596610941215855552, 360.2246176569740362774, 363.3891628020936114035, 366.5533073905408910758, 369.7170619013906999011, 372.8804364485064547049, 376.0434407964318588402, 379.2060843754540293559},
{51.36749420789151010133, 56.88579459032172280786, 61.64300365704914635009, 66.02515595788811672328, 70.1755487534468166833, 74.16608868098592437912, 78.03882448951861094835, 81.82078404082115803534, 85.53050227406132658352, 89.1813054403671036549, 92.7831273559315691097, 96.3435874400613135612, 99.8686667929511262043, 103.3631513220379885347, 106.8309328965159956124, 110.2752203088712330137, 113.6986909039733109102, 117.1036019989525471872, 120.4918743452309732149, 123.8651557112507723331, 127.2248700489723840201, 130.5722560216674299196, 133.908397557241134572, 137.2342483397136241231, 140.5506516339968222858, 143.858356476429508591, 147.1580310052416892564, 150.4502735184540751321, 153.7356217099968716949, 157.0145604334512375042, 160.2875282667837844565, 163.5549230938203161666, 166.8171068741054803008, 170.0744097387389788535, 173.3271335232561461313, 176.5755548278011084986, 179.8199276783767665095, 183.0604858498456037878, 186.2974449008473057071, 189.5310039623248961399, 192.7613473144771611292, 195.9886457813486120695, 199.2130579676716238043, 202.4347313587881248732, 205.6538033013431624974, 208.8704018798359159408, 212.0846467019370515412, 215.2966496036563859577, 218.5065152839088940283, 221.7143418767296255032, 224.9202214682881894262, 228.1242405649178908511, 231.3264805175762325604, 234.5270179074700875629, 237.7259248969921071706, 240.9232695496097382523, 244.119116121912049349, 247.3135253306419817587, 250.5065545972139439053, 253.6982582719315649845, 256.8886878398717919907, 260.0778921101841912047, 263.2659173903639386907, 266.4528076468898570432, 269.6386046534718353865, 272.8233481280223791053, 276.0070758593525878359, 279.1898238244915962557, 282.3716262974387509268, 285.5525159500780913632, 288.7325239459138072671, 291.9116800272221786178, 295.0900125961591375398, 298.2675487903122106167, 301.4443145531405032615, 304.6203346997059603706, 307.795632978062844284, 310.9702321266397438288, 314.1441539279190568359, 317.3174192586924131188, 320.4900481371466092435, 323.6620597670130320549, 326.8334725789940093835, 330.0043042696618268293, 333.1745718380100972722, 336.344291619822594962, 339.5134793200114180533, 342.6821500430642884646, 345.8503183217298169481, 349.0179981440595482455, 352.1852029789164616578, 355.3519458000512518418, 358.5182391088400777174, 361.6840949557704764424, 364.8495249607557339712, 368.0145403323521294102, 371.1791518859480784326, 374.3433700609892475374, 377.5072049372991564115, 380.6706662505505944975},
{52.41424545316938954605, 57.96479573260000961454, 62.74672269292185688982, 67.14950650641455354162, 71.31780130996581383524, 75.3242315676269617312, 79.21128672343311257086, 83.00629105264682460526, 86.72799160321519328262, 90.38987336336522750364, 94.0019930394219562851, 97.5720677442040484051, 101.1061579105966754003, 104.6091150015805254142, 108.0848858505062284818, 111.5367259040976448245, 114.9673525273887043327, 118.3790576816765052042, 121.7737923489996301747, 125.1532308634149315927, 128.5188206682256459071, 131.8718213160297505961, 135.2133354039745810888, 138.5443333773484483655, 141.8656736118151126887, 145.1781188181398389659, 148.4823495522371164542, 151.7789754247097266192, 155.0685444658481943398, 158.3515509995687478097, 161.6284423028923365634, 164.8996242692967915616, 168.1654662496741475848, 171.4263052101779455408, 174.6824493194142259512, 177.9341810563654769295, 181.1817599137769159657, 184.425424758466415762, 187.6653958993833831368, 190.9018769056632190493, 194.1350562099643841794, 197.3651085266979103224, 200.5921961101039848156, 203.8164698732941550443, 207.0380703862017797399, 210.2571287677422493518, 213.4737674852788150952, 216.6881010726403594508, 219.9002367763806002209, 223.1102751386528954476, 226.3183105239596168101, 229.5244315960863397412, 232.7287217507213928007, 235.9312595085681009059, 239.1321188731618160423, 242.3313696570912459122, 245.529077779880971699, 248.7253055404088282969, 251.920111866399185339, 255.1135525432437284414, 258.3056804241488905615, 261.4965456233883922549, 264.6861956942459953688, 267.8746757930638094822, 271.0620288306621287044, 274.2482956122651036886, 277.4335149669502506735, 280.6177238675368797457, 283.8009575417372830935, 286.9832495753134935023, 290.1646320079103383239, 293.3451354221712809975, 296.5247890266862137184, 299.7036207332691190111, 302.8816572290176399671, 306.0589240435654659573, 309.2354456119015091192, 312.4112453330966414797, 315.5863456252488667535, 318.7607679769308492425, 321.9345329953993936215, 325.10766045180448093, 328.2801693236155684184, 331.4520778344648337426, 334.6234034915906925627, 337.7941631210500702775, 340.9643729008544103325, 344.1340483921721172626, 347.303204568728941396, 350.4718558445276061801, 353.6400160999986622801, 356.8076987066860391111, 359.9749165505629775394, 363.1416820540668984167, 366.3080071969352288139, 369.4739035359182160223, 372.6393822234392589493, 375.804454025268232803, 378.9691293372686357217, 382.1334182012751090401},
{53.4603471781673746932, 59.04273880140359825593, 63.84909553552081717953, 68.27228855396657405842, 72.45830681617040625042, 76.48048072319848024524, 80.38173301291464283987, 84.18967934900254771861, 87.92327527257988629593, 91.59616179765189233176, 95.2185164237728759313, 98.798152296457591701, 102.3412078407573711741, 105.852598976471843446, 109.3363265890797402774, 112.7956920143877399924, 116.2334519898217348426, 119.6519325681580541644, 123.0531144872862204463, 126.4386982366174039104, 129.8101543944113318696, 133.1687630915549125108, 136.5156453218400235718, 139.8517880520131116913, 143.1780645568037264206, 146.4952510339644409907, 149.8040402906562399732, 153.1050531018929934002, 156.3988477021050331271, 159.685927767300591675, 162.9667491676007511696, 166.2417257110184115888, 169.5112340542602264802, 172.7756179214991869765, 176.0351917449317840549, 179.2902438196288689828, 182.5410390483371451705, 185.7878213384652405719, 189.0308157027267028624, 192.2702301062311154266, 195.5062570957708629665, 198.7390752413045453677, 201.9688504149253278807, 205.1957369287185301518, 208.4198785496967388821, 211.6414094073259154676, 214.8604548069218310311, 218.0771319603225372803, 221.2915506436652241988, 224.5038137907629741569, 227.7140180294467157986, 230.9222541672760633719, 234.1286076322019004134, 237.333158873060741281, 240.5359837241773219316, 243.7371537378320340461, 246.9367364879008718473, 250.1347958475868133658, 253.331392243824076942, 256.526582890642998146, 259.7204220035270752225, 262.9129609965697335519, 266.1042486640420818741, 269.294331347810577693, 272.4832530918918583803, 275.6710557852982768888, 278.8577792942095560013, 282.0434615844014292109, 285.2281388347694409285, 288.4118455427037460508, 291.5946146219974958831, 294.776477493906114905, 297.9574641719165002421, 301.1376033407330791323, 304.3169224299410140346, 307.4954476827650184529, 310.6732042203046874618, 313.8502161015934729215, 317.0265063797980218158, 320.2020971548471761369, 323.3770096227551781919, 326.5512641218812490198, 329.7248801763474565698, 332.8978765368184407431, 336.0702712188299164922, 339.2420815388377586128, 342.4133241481457271574, 345.5840150648573823643, 348.7541697039863397372, 351.9238029058486199096, 355.092928962851356526, 358.2615616447834513229, 361.4297142227058308936, 364.5973994915316944192, 367.7646297913804833381, 370.9314170277831963215, 374.0977726908110655298, 377.2637078731944575487, 380.4292332874941237484, 383.5943592823825632198},
{54.5058216528616378325, 60.11965897261044970681, 64.95016574487193850018, 69.39355160016182821892, 73.5971191185893758302, 77.63489321145895037116, 81.55022279803500902652, 85.37101010078500028238, 89.11641567794534322331, 92.80023396146570054381, 96.4327612261231223378, 100.0219050507433249803, 103.5738805606468760632, 107.0936670727300331798, 110.5853186441898872642, 114.0521817606197540798, 117.4970519053676807194, 120.9222886876575016281, 124.3299021411716703527, 127.7216185127187346512, 131.0989311694567126692, 134.463140518218537676, 137.8153856841788088163, 141.1566699219318750951, 144.4878811988329380593, 147.8098090166243723455, 151.1231582710368873776, 154.4285607565081529673, 157.7265847820799635319, 161.0177432598823019191, 164.3025005490964368475, 167.5812782787605023979, 170.8544603272066135328, 174.1223971007100456084, 177.3854092265000003568, 180.6437907537406095446, 183.8978119390497938849, 187.1477216795484908625, 190.3937496455478271881, 193.6361081562000659354, 196.8749938343129556226, 200.1105890707124994478, 203.3430633237700077786, 206.5725742757782851688, 209.7992688646064249179, 213.0232842063547820715, 216.2447484224696246391, 219.463781382879666117, 222.6804953751191863774, 225.8949957080524127754, 229.1073812576688969964, 232.3177449614453446042, 235.526174266937610932, 238.7327515395543106803, 241.9375544338517182746, 245.1406562321626773552, 248.3421261539170830057, 251.5420296386173236817, 254.7404286050898308855, 257.9373816893360093954, 261.1329444630459438665, 264.327169634611032945, 267.5201072342725531704, 270.7118047848682557283, 273.9023074594851798516, 277.0916582271911403841, 280.2798979878974321655, 283.4670656972991503432, 286.6531984827454016827, 289.8383317508080697208, 293.0224992872433956783, 296.2057333499743231927, 299.388064755662354244, 302.5695229603847307591, 305.7501361348853560183, 308.9299312348253613885, 312.1089340664210476816, 315.2871693478225972616, 318.4646607665560341974, 321.641431033323028976, 324.8175019324279707233, 327.9928943690789721926, 331.167628413788874011, 334.3417233440836478225, 337.5151976837086622121, 340.6880692395078926962, 343.8603551361371697443, 347.0320718487598264698, 350.203235233861505486, 353.3738605583103012536, 356.5439625267787508582, 359.7135553076353539304, 362.8826525574052221164, 366.0512674438920591566, 369.2194126680468906236, 372.3871004846627406079, 375.5543427219687398196, 378.7211508001918995374, 381.8875357491499570371, 385.0535082249342530839},
{55.55068994576704597839, 61.19558957662027154356, 66.04997465178777815709, 70.51334266959210594905, 74.73428943022045700673, 78.78752336392689211902, 82.71681273066914547963, 86.55034166609236067173, 90.30747240105179859433, 94.0021502752785439368, 97.6447883953319917475, 101.2433872321126136097, 104.8042373654192337299, 108.3323804865680646629, 111.831922974153274003, 115.3062557489142142849, 118.7582124340043322992, 122.1901856769683016389, 125.6042143613372773476, 129.0020501051945963798, 132.3852087284971500637, 135.7550106200862730159, 139.1126127783685147572, 142.4590345182031669591, 145.7951782982465449334, 149.1218467455789144982, 152.4397566855168525326, 155.7495507900724439636, 159.0518073160352128591, 162.347048297946625403, 165.6357464819214897673, 168.9183312261290119379, 172.195193547697344722, 175.4666904602262070874, 178.7331487183690579557, 181.9948680641734990145, 185.2521240526420137215, 188.5051705202506058203, 191.7542417491568128117, 194.9995543709480080754, 198.2413090465736383682, 201.4796919532235709951, 204.7148761040901356273, 207.9470225229742511647, 211.1762812924019966452, 214.4027924911775899131, 217.6266870350092377888, 220.8480874319237789906, 224.0671084625687698188, 227.283857794133743904, 230.4984365354629767148, 233.7109397399453744359, 236.9214568619246437303, 240.1300721716513617316, 243.3368651331787543408, 246.5419107490700386784, 249.7452798753239320497, 252.947039509525421159, 256.147253054881973307, 259.3459805625033908041, 262.5432789540200181824, 265.7392022264035774795, 268.9338016406529374662, 272.1271258958297242517, 275.3192212897725425763, 278.5101318676808771223, 281.6998995596380694887, 284.8885643080350527876, 288.0761641857609976965, 291.2627355059421538875, 294.4483129239346414719, 297.6329295322096193363, 300.8166169487091441222, 303.9994053991972783952, 307.1813237940828640159, 310.3623998001471973968, 313.5426599075710600875, 316.7221294926206749314, 319.9008328763207401565, 323.0787933794143596436, 326.2560333738841017373, 329.4325743312852858414, 332.6084368681216550322, 335.7836407884746137421, 338.9582051240799884784, 342.1321481720306259924, 345.3054875302689171047, 348.4782401310203831541, 351.6504222723076590862, 354.8220496476734396683, 357.9931373742311224949, 361.163700019152893379, 364.3337516246967759383, 367.5033057318666356342, 370.6723754027922245938, 373.8409732419100192541, 377.0091114160197861013, 380.1768016732864646572, 383.3440553612520391586, 386.5108834439175430282},
{56.59497201148920453306, 62.27056222938330852222, 67.14856151247427362705, 71.63170647973563702295, 75.86986650546911903348, 79.93842295772361694617, 83.88155685243770149528, 87.72772976632766699226, 91.49650238250814517645, 95.20196853060236330162, 98.85465627604994061828, 102.4626574956775824763, 106.0323370217186335086, 109.5687979324418241392, 113.0761981061645630432, 116.5579722076499343602, 120.0169914131919267875, 123.4556809067186935185, 126.8761079892106596978, 130.2800492752857238647, 133.6690427112008051087, 137.0444283819906164388, 140.4073809080323158034, 143.758935442722341738, 147.1000087396773108321, 150.4314163769099420142, 153.7538869539592867229, 157.0680738816496466583, 160.3745652402861260099, 163.6738920753527436695, 166.9665354196775040705, 170.2529322702846151509, 173.533480701638562911, 176.8085442610410183683, 180.0784557639313299079, 183.343520584840562167, 186.6040195223401204481, 189.8602113024545874297, 193.1123347738832541817, 196.3606108393971631412, 199.6052441604915599504, 202.8464246664264821003, 206.0843288939091588223, 209.3191211796493294282, 212.5509547246865821691, 215.7799725466164993633, 219.0063083335259493738, 222.2300872115044880766, 225.4514264359621353627, 228.6704360156002884363, 231.8872192767089244843, 235.1018733744643214566, 238.3144897570485322116, 241.5251545876812029249, 244.7339491290266162514, 247.9409500938980171608, 251.1462299657140344236, 254.349857291757263725, 257.551896951933575184, 260.7524104054246940352, 263.9514559173595599463, 267.1490887673964019538, 270.3453614419027249415, 273.5403238112405491055, 276.7340232935059250539, 279.9265050059321073672, 283.1178119050423669736, 286.3079849165291655837, 289.4970630557395069922, 292.6850835395601736665, 295.8720818904199183345, 299.0580920330573555451, 302.2431463846422877782, 305.4272759387836367351, 308.610510343908277947, 311.7928779764512377239, 314.974406009258331096, 318.1551204755668950371, 321.3350463288983623532, 324.5142074991676413891, 327.6926269452882750515, 330.8703267045288500071, 334.0473279388548490113, 337.2236509784708526772, 340.3993153627604949644, 343.5743398788056761762, 346.7487425976520757829, 349.9225409084748405429, 353.0957515507863226519, 356.2683906448167934923, 359.4404737201890588082, 362.612015742998759494, 365.7830311414037770114, 368.953533829818500595, 372.1235372318016893049, 375.29305430172021676, 378.4620975452650671408, 381.6306790388905101728, 384.7988104482423774348, 387.9665030456367538034},
{57.63868677030260995343, 63.3446069517857320521, 68.2459636497030823857, 72.74868559458433838794, 77.00389680050369194133, 81.08764137887855838471, 85.04450675837903735534, 88.90322764843931940118, 92.6835600813431480381, 96.39974404597633670485, 100.06242076054417804501, 103.679772073880920588, 107.2582359101498435895, 110.8029757805897082792, 114.3182002688573673489, 117.8073871150605753016, 121.2734444805658115113, 124.7188295992546475833, 128.1456377701574764923, 131.5556702406339629319, 134.9504867659960524546, 138.3314468495039181806, 141.6997424889156742617, 145.0564244601215149321, 148.4024236202088205659, 151.7385683278626396489, 155.0655988050260028808, 158.3841790656057862273, 161.6949068917846011657, 164.9983222307379092249, 168.2949143036807655983, 171.585127657831574056, 174.8693673449058537821, 178.1480033734508237669, 181.421374554039018814, 184.6897918341141759998, 187.9535412016944063727, 191.2128862231218556377, 194.4680702688058660947, 197.7193184718335308789, 200.9668394569563117753, 204.2108268714496473663, 207.4514607444098926034, 210.6889086969857694436, 213.9233270226721823283, 217.1548616539905649409, 220.3836490295369993525, 223.6098168734135299264, 226.833484897402307483, 230.0547654348423975151, 233.2737640139815145938, 236.4905798775640200912, 239.7053064545531788867, 242.9180317891460768614, 246.1288389316041281206, 249.3378062948745267338, 252.5450079805048648172, 255.7505140769432437242, 258.9543909329601897047, 262.1567014086187071341, 265.3575051059482683539, 268.5568585812418900555, 271.7548155406879806815, 274.9514270208663745283, 278.146741555477505712, 281.3408053295321282508, 284.5336623221038863387, 287.7253544386362642177, 290.9159216336971793548, 294.10540202498716031, 297.2938319993293209139, 300.4812463113000370826, 303.6676781750973396359, 306.8531593501886769952, 310.0377202212301110287, 313.2213898727045210706, 316.4041961586864232179, 319.5861657681050551355, 322.7673242858449844575, 325.9476962499942795744, 329.1273052055238905322, 332.3061737546580213686, 335.484323604173665646, 338.6617756098478875446, 341.8385498182536522226, 345.0146655060888556037, 348.1901412172085105807, 351.3649947975166673946, 354.5392434278624503592, 357.7129036550734650021, 360.8859914212496654603, 364.0585220914314790075, 367.2305104797474800623, 370.401970874139115743, 373.5729170597528424652, 376.7433623410834784869, 379.9133195629465570097, 383.082801130351930141, 386.2518190273457821828, 389.4203848348835221199},
{58.68185218062083653775, 64.41775227862667918544, 69.34221658194400106368, 73.86432056543881685684, 78.13642462048113657147, 82.23522577251964094103, 86.2057117477212626438, 90.07688623460629922691, 93.86869762242819100315, 97.5955298113020146345, 101.26813542937559826548, 104.8947849131315621666, 108.4819881576291088042, 112.0349681849596029676, 115.5579835157464835513, 119.0545543181900009347, 122.5276251884465948147, 125.9796849387789039449, 129.412856459347112512, 132.8289652769892685014, 136.2295926477461406205, 139.6161172227146876068, 142.9897481389027826018, 146.3515515841667186676, 149.7024723322941599324, 153.0433513564253354143, 156.3749403525558392502, 159.697913804920357925, 163.0128790784938813202, 166.3203849150824731037, 169.6209286278398751121, 172.9149622271258368678, 176.2028976631961350416, 179.4851113345593700979, 182.7619479822652428889, 186.0337240679435064936, 189.3007307156481966595, 192.5632362834039078515, 195.8214886189935757334, 199.0757170453602306975, 202.3261341135529705424, 205.5729371550721579837, 208.8163096604836269658, 212.0564225070606746419, 215.2934350548065973375, 218.527496127375999502, 221.7587448920441465987, 224.987311650885753122, 228.2133185536500437689, 231.4368802414031026039, 234.6581044288072279461, 237.8770924318842883177, 241.0939396472365611047, 244.3087359879501441976, 247.521566280762927319, 250.7325106285248985438, 253.9416447414996403243, 257.1490402406409133442, 260.3547649356177751921, 263.5588830800478052337, 266.7614556061240410107, 269.9625403405815535439, 273.162192203739444719, 276.3604633931694072078, 279.5574035533794168699, 282.7530599327577052744, 285.9474775288953862619, 289.1406992232938432912, 292.3327659063633836716, 295.5237165935311474546, 298.7135885331974557355, 301.9024173072095132007, 305.0902369244586185091, 308.2770799081508938736, 311.462977377251247991, 314.6479591225551594522, 317.8320536778023216124, 321.0152883862097089235, 324.1976894627687569308, 327.3792820526216966284, 330.5600902858052995948, 333.7401373286260657784, 336.91944543190894966, 340.0980359763418329956, 343.2759295151199012401, 346.4531458140776779703, 349.6297038894815502074, 352.8056220436420290925, 355.9809178984926027998, 359.1556084272707341529, 362.3297099844262290936, 365.5032383338727600436, 368.6762086756896863243, 371.8486356713733969132, 375.0205334677301414516, 378.1919157194956529966, 381.3627956107607460726, 384.5331858752764471291, 387.7030988157070373992, 390.8725463218946205769},
{59.72448530511612574401, 65.49002535827049444521, 70.43735414168479280761, 74.97865006015305766596, 79.26749225493004328991, 83.3812211812097245911, 87.36521896297228979174, 91.24875426053259080581, 95.05196493287701722372, 98.78937662157046770454, 102.47185168190943893709, 106.1077478007280349715, 109.7036457604786746479, 113.2648272023302904146, 116.7955998403052167503, 120.2995256439098641771, 123.779585110823926414, 127.2382981743579409159, 130.6778149208627653324, 134.0999848135194811981, 137.5064103093669760757, 140.8984889442696425965, 144.2774467626041138273, 147.6443651590148661075, 151.0002026418071040601, 154.345812636299249165, 157.6819581675757847817, 161.0093240603593368498, 164.3285271458410613879, 167.6401248555561836691, 170.9446225000065184989, 174.2424794672371499526, 177.5341145287040562768, 180.819910402767464327, 184.1002176973019420128, 187.3753583302517093152, 190.6456285090215106508, 193.9113013352958125929, 197.1726290904085284112, 200.4298452471261212804, 203.6831662461889774892, 206.9327930698182374175, 210.1789126393582764427, 213.4216990600708955039, 216.6613147326551742059, 219.8979113482020773032, 223.1316307808983453521, 226.3626058907846252056, 229.5909612471797881033, 232.8168137819518047424, 236.0402733806007524922, 239.2614434180852065554, 242.4804212454397503786, 245.6972986324742982498, 248.9121621711953158518, 252.1250936440282867781, 255.3361703604361598969, 258.5454654651085796502, 261.7530482205318877601, 264.9589842664321764663, 268.1633358583063321823, 271.3661620870133516967, 274.5675190811854342809, 277.7674601940313754325, 280.9660361759401440845, 284.1632953341472580845, 287.3592836805981547545, 290.5540450690290179264, 293.7476213221846088933, 296.9400523500029537012, 300.1313762595168801811, 303.3216294571511790544, 306.5108467440305477543, 309.6990614048565630983, 312.8863052908609362904, 316.0726088972965480064, 319.2580014358866459881, 322.4425109026155906659, 325.626164141211198255, 328.808986902638653719, 331.9910039008987945498, 335.1722388653989890787, 338.3527145901425752719, 341.5324529799626448078, 344.7114750940076377782, 347.8898011866695650454, 351.0674507461305290166, 354.2444425306894190035, 357.4207946030180805996, 360.5965243624847803359, 363.7716485756723008086, 366.9461834052084123615, 370.1201444370176901363, 373.2935467060956036055, 376.4664047208984311864, 379.638732486435783784, 382.810543526146302912, 385.9818509026313817144, 389.1526672373164959865, 392.3230047291048868021},
{60.76660237115231651409, 66.56145204392723258024, 71.53140858402063884114, 76.09171098196227944754, 80.39714010243089581241, 84.52567067254967980716, 88.52307351841248325404, 92.41887840338995882573, 96.23340986841095579198, 99.98133320091686894354, 103.67361885754148668846, 107.3187104828984109387, 110.9232586990425125467, 114.4926029033555393231, 118.031099283359076769, 121.5423510026362352819, 125.0293739434088162598, 128.4947187163524368852, 131.9405622205746540734, 135.3687775222071708244, 138.7809879878401945777, 142.1786097821037279656, 145.5628856309094332628, 148.9349119356989432607, 152.2956607615701123926, 155.6459978275800285299, 158.9866973462458896974, 162.3184543557916150843, 165.6418950395109364281, 168.9575854158926302341, 172.2660387000306789553, 175.5677215737819865672, 178.8630595538262299905, 182.1524416094386542191, 185.4362241526736886905, 188.7147345007810945278, 191.9882738925673850141, 195.2571201259801378928, 198.5215298726107879175, 201.781740715461249717, 205.0379729487272224578, 208.2904311721518611184, 211.5393057074155597845, 214.7847738598310281708, 218.0270010451352294514, 221.266141798275043176, 224.5023406786637698887, 227.7357330843546759031, 230.9664459858666141944, 234.1945985889496804699, 237.4203029343507922745, 240.6436644415933440231, 243.8647824028917349349, 247.0837504325560148516, 250.300656876583904773, 253.5155851865702912645, 256.7286142615740818054, 259.9398187611574797793, 263.1492693924436330497, 266.3570331737171314124, 269.5631736768111667017, 272.7677512502795821414, 275.9708232251366668992, 279.1724441047582770603, 282.3726657403711803959, 285.5715374934104402757, 288.7691063858946182034, 291.9654172398533943163, 295.1605128067399988874, 298.3544338876699948885, 301.5472194452470533968, 304.7389067076642103981, 307.9295312657046368637, 311.1191271632082832544, 314.3077269815190823765, 317.4953619183810198414, 320.6820618617097057017, 323.867855458628575937, 327.0527701801250552986, 330.2368323816515169208, 333.4200673599683213605, 336.6024994065012931445, 339.7841518574634183432, 342.9650471409700762986, 346.1452068213585348011, 349.3246516409055478092, 352.5034015591215269734, 355.6814757897857605005, 358.8588928358743897351, 362.0356705225212044142, 365.2118260281406740188, 368.3873759138328986318, 371.562336151181251955, 374.7367221485453243834, 377.9105487759442859741, 381.0838303886189155374, 384.2565808493542269712, 387.4288135496388169982, 390.6005414297317142654, 393.7717769977025874153},
{61.80821882611492380223, 67.6320569774018007451, 72.62441068647011377379, 77.2035385788987006031, 81.5254067856069601269, 85.66861545704577451943, 89.67931861895765198901, 93.5873034003381972301, 97.41307833057519587753, 101.17144631784211732791, 104.87348434843277230907, 108.52772077470355897516, 112.1408750445248730866, 115.7183434761890823247, 119.2645300334125729782, 122.7830784853226314378, 126.2770395972946651097, 129.7489942267764025975, 133.201145713248684792, 136.6353904017751289547, 140.0533722850343541921, 143.4565259072440984342, 146.8461104558672349458, 150.2232371441790609072, 153.5888914206750298745, 156.94395114344673354, 160.2892015740147803232, 163.6253478399081443018, 166.9530253648241393335, 170.2728086535195763938, 173.5852187347338789956, 176.8907295018276947847, 180.1897731420814125236, 183.4827448079176036563, 186.770006653933404687, 190.0518913405423050869, 193.3287050867476253034, 196.6007303399994173192, 199.8682281193947454142, 203.1314400790415885323, 206.3905903307407306508, 209.6458870588803372618, 212.8975239552998325641, 216.1456814976413642239, 219.3905280911945840049, 222.6322210913163986332, 225.8709077210628109003, 229.1067258966180493861, 232.3398049713771575444, 235.5702664080758849475, 238.7982243871205876001, 242.0237863582138803521, 245.2470535414687467451, 248.4681213834298864429, 251.6870799727558296312, 254.9040144197418731525, 258.119005203368171876, 261.3321284891276733476, 264.543456420515256607, 267.7530573867342474519, 270.9609962688925537293, 274.1673346667121934069, 277.3721311075580670095, 280.5754412394002907004, 283.7773180091557149179, 286.9778118277053852052, 290.1769707227530737164, 293.3748404805734077934, 296.5714647775946436926, 299.7668853026691407435, 302.9611418708026699449, 306.1542725290406181588, 309.3463136551438644621, 312.5373000496286879977, 315.7272650216927154138, 318.9162404695019318648, 322.1042569552715520644, 325.2913437755355427352, 328.4775290269653365169, 331.6628396680673685658, 334.8473015770611387555, 338.0309396062142356397, 341.2137776328878716872, 344.3958386075257239479, 347.5771445988000299843, 350.7577168361117603461, 353.9375757496261026955, 357.1167410080102947683, 360.2952315540278964542, 363.4730656381317730508, 366.6502608501872629962, 369.8268341494471264983, 373.0028018928908291183, 376.178179862032428267, 379.3529832882937305362, 382.5272268770324107645, 385.7009248303083730482, 388.8740908684657385212, 392.0467382506024186781, 395.2188797939942348301},
{62.84934938815283691535, 68.70186366605437713176, 73.71638984086567210534, 78.31416654468739191892, 82.65232925732739488467, 86.81009499713138666266, 90.83399567025756967756, 94.75407215845514485046, 98.59101437560090024756, 102.35976089235643299297, 106.07149370446698929171, 109.73482466247742953196, 113.3565410586857725242, 116.9420953232862908779, 120.4959385204682340018, 124.0217544542536652387, 127.5226282867181709482, 131.0011707040449716568, 134.4596111243212418638, 137.8998688565436722297, 141.3236082437116488427, 144.7322819670413698938, 148.1271654612213947522, 151.5093845612691431986, 154.8799379298866633223, 158.2397154131301523936, 161.5895131862406105527, 164.9300463445827211905, 168.2619594429236877232, 171.5858353736554889679, 174.9022028899978786462, 178.2115430160532804671, 181.5142945364208481252, 184.8108587200649240866, 188.1016034034939905631, 191.3868665350124101213, 194.666959263364382656, 197.9421686393856052572, 201.2127599874782629105, 204.4789789941971437513, 207.7410535534965409391, 210.9991954018683601284, 214.2536015714143948334, 217.5044556846162849384, 220.7519291110197103575, 223.9961820030963840417, 227.2373642260784726101, 230.4756161944874753597, 233.7110696263329788033, 236.9438482244793636699, 240.1740682934245456001, 243.4018392986668142847, 246.6272643749232685102, 249.8504407886811779123, 253.0714603598912031865, 256.2904098470317143988, 259.5073712992723033535, 262.722422379030199273, 265.9356366578358139405, 269.1470838880948022527, 272.3568302530468795821, 275.564938596970328156, 278.7714686374606929435, 281.9764771614184070611, 285.1800182062094191858, 288.3821432273122688103, 291.5829012536318595142, 294.7823390315421784745, 297.9805011586154798126, 301.1774302079023343671, 304.3731668435440199675, 307.5677499284247479454, 310.761216624505121765, 313.9536024864190713357, 317.1449415488634915371, 320.3352664082622291518, 323.5246082991432914558, 326.7129971656296518197, 329.9004617284093290832, 333.0870295475191020511, 336.2727270812479221267, 339.4575797414404832383, 342.641611945458214059, 345.8248471650339213686, 349.0073079722372123868, 352.1890160827504604398, 355.3699923966392769575, 358.5502570367870575486, 361.7298293851500418655, 364.9087281169773423368, 368.0869712331294450461, 371.2645760906186682933, 374.4415594314858922275, 377.6179374101194671828, 380.7937256191144978711, 383.9689391137636215672, 387.1435924352638934371, 390.3176996327184095544, 393.491274284005791336, 396.6643295155855821187},
{63.89000809278606141106, 69.77089455363011438749, 74.80737413807146651834, 79.42362711191672985933, 83.77794289892726243698, 87.95014710813874782081, 91.98714438080497124971, 95.91922585682338851108, 99.76726031562773910799, 103.54632009572487300335, 107.26769073107503277436, 110.94006639941323040405, 114.5703012869673796008, 118.163903151922685868, 121.7253695038454935894, 125.2584236281159495525, 128.7661846113673236041, 132.2512925625302465003, 135.7160026267332911792, 139.1622567705883017916, 142.5917394190655680564, 146.0059211541519034781, 149.4060934489085331723, 152.7933965747237910586, 156.168842243396058945, 159.5333321414107309728, 162.8876732255122857055, 166.2325904400948311593, 169.5687373639768946799, 172.8967051805666519886, 176.2170302801514892158, 179.5302007383385328034, 182.836661865090256458, 186.1368209804603896429, 189.431051543238870517, 192.7196967352165741611, 196.0030725851744186978, 199.2814707018660916856, 202.5551606733572987408, 205.824392180469795901, 209.0893968642691590042, 212.35038998115731949, 215.6075718738948098262, 218.8611292825577169115, 222.1112365158534955346, 225.3580565002383320307, 228.6017417217857829215, 231.8424350736634559959, 235.0802706203105711848, 238.3153742879171165865, 241.5478644895386572737, 244.7778526921019556704, 248.0054439316346123962, 251.2307372822616555719, 254.4538262838325696897, 257.6747993324564416785, 260.8937400377164195439, 264.1107275498956243366, 267.3258368601650814184, 270.5391390763518016358, 273.7507016766148322941, 276.9605887431029940816, 280.1688611774451165315, 283.3755768997276332319, 286.5807910324417846965, 289.7845560707303187764, 292.9869220401288393411, 296.1879366428775735553, 299.3876453937733642405, 302.586091746437473686, 305.783317210790864799, 308.9793614624537553691, 312.1742624447193384154, 315.3680564636916852261, 318.5607782771241788407, 321.75246117744665015, 324.9431370694260824585, 328.1328365428667670331, 331.3215889407206528615, 334.5094224229469178327, 337.6963640264311256344, 340.8824397212483966051, 344.0676744635315235887, 347.252092245183650948, 350.4357161406557808714, 353.6185683509917760925, 356.8006702453275146538, 359.9820424000162625572, 363.1627046355390235409, 366.3426760513464765022, 369.5219750587680083883, 372.700619412113193717, 375.8786262380817717602, 379.0560120635896486339, 382.2327928421106321386, 385.4089839786264275479, 388.5846003532708245483, 391.759656343747936859, 394.9341658465987695698, 398.1081422973852423838},
{64.9302083357818975518, 70.83917108554209107751, 75.89739044619869743622, 80.53195113819165452003, 84.90228161116330340974, 89.08880805193328455971, 93.13880285675041427833, 97.08280404144513753795, 100.9418568129304030987, 104.73116544342819135372, 108.46211758050897521631, 112.14348859484657939301, 115.7821986455716130896, 119.3838100589196634311, 122.9528661544609893055, 126.4931291617802277804, 130.0077516336442221585, 133.4994027073080819426, 136.9703629131832102445, 140.4225965775350313932, 143.8578079461060393697, 147.2774852715681600868, 150.6829358617948474346, 154.0753142437469624996, 157.4556450171691035501, 160.8248415648767544092, 164.1837214958875863821, 167.5330194874180806177, 170.8733980375839317567, 174.205456526164712321, 177.5297388948246547709, 180.8467401929406476531, 184.1569121851932725375, 187.460668178415605266, 190.7583871950443474756, 194.0504175968185495152, 197.3370802436041487161, 200.6186712572576663325, 203.8954644484311304217, 207.1677134545201113329, 210.4356536290773141236, 213.6995037165784304584, 216.9594673411429546217, 220.2157343334528255034, 223.4684819164975307169, 226.7178757677647974406, 229.9640709729793590703, 233.2072128843792748416, 236.4474378947382931989, 239.6848741368360685149, 242.9196421167989120651, 246.1518552886441414428, 249.3816205764299021101, 252.6090388496140626094, 255.8342053565394169829, 259.0572101203705860384, 262.2781383012962693652, 265.4970705283668483723, 268.7140832039517363271, 271.9292487834648954697, 275.1426360327135124752, 278.3543102649679676483, 281.5643335596258953196, 284.7727649641450238195, 287.97966068074495378, 291.1850742392239702634, 294.3890566571007248516, 297.5916565881698857425, 300.7929204604536753766, 303.9928926044359064497, 307.1916153723802295747, 310.3891292494585572669, 313.5854729573479351225, 316.7806835508935414041, 319.9747965083811832333, 323.1678458159138994931, 326.3598640463434467006, 329.5508824331679824689, 332.7409309397716876039, 335.9300383243499570928, 339.1182322008347680211, 342.3055390961085689883, 345.4919845037712397229, 348.6775929347030839188, 351.862387964647214895, 355.0463922790168703983, 358.2296277151169702437, 361.4121153019544488977, 364.5938752977984124231, 367.774927225638858685, 370.9552899066814482014, 374.1349814920055193591, 377.3140194925031153463, 380.4924208072081499483, 383.6702017501169124908, 386.847378075593833306, 390.0239650024497413246, 393.1999772367736918045, 396.375428993593777088, 399.5503340174371135562},
{65.96996291265656964242, 71.9067137691271778468, 76.98646448291579247325, 81.63916818590287505231, 86.02537789854964523973, 90.22611262385002817155, 94.2890076900480949007, 98.24484471358996435208, 102.11484296772964666393, 105.91433688189333376759, 109.65481483709250892595, 113.3451322977351752807, 116.9922745029613054273, 120.6018576100232684552, 124.1784701319904858791, 127.7259127211903748282, 131.2473709512549369355, 134.7455426044457911958, 138.2227332641272910864, 141.6809293263024771986, 145.121854603182451881, 148.5470147939699489573, 151.9577328429087355137, 155.35517735616331504, 158.7403856641167885816, 162.1142827051555319735, 165.4776966142482210633, 168.8313716877623376807, 172.1759792405700187345, 175.5121267561112014621, 178.8403656434265854643, 182.1611978494059984422, 185.4750815240954392812, 188.7824358979272244971, 192.0836454993373712268, 195.3790638173365762175, 198.669016494675472697, 201.9538041221522806997, 205.2337046924962481159, 208.508975762475781719, 211.7798563639315977595, 215.0465686979424954238, 218.3093196410001400503, 221.5683020876700817055, 224.8236961505691159922, 228.07567023545194282, 231.3243820066601202715, 234.5699792560535390421, 237.8126006867468516825, 241.0523766214522656293, 244.2894296439386939734, 247.5238751810170822911, 250.7558220315214360639, 253.9853728479489286559, 257.2126245757292739976, 260.4376688544957676498, 263.6605923852134811437, 266.8814772665719144711, 270.1004013036598379608, 273.3174382916005911918, 276.5326582765296062595, 279.7461277960363557472, 282.9579101009651897963, 286.1680653602692920123, 289.3766508504355665312, 292.5837211308425254428, 295.789328206275488784, 298.9935216777013336549, 302.1963488822966537949, 305.3978550236268098189, 308.5980832927874900062, 311.7970749812437874021, 314.9948695860333257832, 318.1915049079386745281, 321.3870171431793446722, 324.5814409691243262692, 327.7748096244817734911, 330.9671549843825108449, 334.1585076307380322975, 337.3488969182211652013, 340.5383510361881945297, 343.7268970668346567228, 346.9145610398529221301, 350.1013679838378304325, 353.2873419746667941114, 356.4725061810627366876, 359.6568829075318036073, 362.8404936348528129226, 366.023359058281756582, 369.2054991236221930511, 372.3869330613009736883, 375.5676794185783164951, 378.7477560900116899699, 381.9271803462842147341, 385.1059688615002577199, 388.2841377390435168595, 391.4617025360861138804, 394.638678286830975426, 397.815079524564039994, 400.9909203025875365511},
{67.00928405511982077769, 72.97354222933793721495, 78.07462088238750225372, 82.74530659617891498095, 87.14726294765078462677, 91.36209423350656777529, 95.43779404049553193008, 99.40538441212085834822, 103.28625640011241075313, 107.09587286949520008275, 110.84582159692556362151, 114.54503707478846387062, 118.20056875621435751827, 121.8180859143419029197, 125.4022216572955108024, 128.9568145537204407701, 132.4850827654666448111, 135.9897523471510791748, 139.4731536118293332029, 142.9372947430742709442, 146.3839188719112795145, 149.8145489256474282476, 153.2305232914051549024, 156.6330244824741421315, 160.0231024062959181223, 163.4016934193140307962, 166.7696360589566873634, 170.1276841295436211507, 173.4765176623251221706, 176.8167521535833982034, 180.1489463973933904973, 183.473609163354147821, 186.791204918798017518, 190.1021587556926455073, 193.4068606518037718144, 196.705669171593295146, 199.998914693244954623, 203.2869022329914980245, 206.5699139257004885408, 209.8482112108082682485, 213.1220367646748724789, 216.3916162138837837526, 219.6571596586326490027, 222.9188630309231639204, 226.1769093085787865648, 229.4314696030545479634, 232.6827041364402811912, 235.9307631209063440431, 239.175787552026557347, 242.4179099258779196038, 245.6572548885131357937, 248.8939398252914173885, 252.1280753966037462674, 255.3597660257148933006, 258.5891103437445585625, 261.8162015962063680675, 265.0411280150014400013, 268.263973159310591432, 271.4848162284357717401, 274.7037323492984089347, 277.9207928410028282391, 281.1360654586106640729, 284.3496146180420877382, 287.5615016038173440649, 290.7717847611738127191, 293.9805196739364114734, 297.1877593293799260222, 300.3935542711984572246, 303.5979527415876221972, 306.8010008133477076723, 310.0027425128291650135, 313.2032199344643691327, 316.4024733475603222319, 319.60054129596499806, 322.7974606911644489989, 325.99326689931790033, 329.187993822693189917, 332.3816739759245138173, 335.5743385574780148124, 338.7660175166778673621, 341.9567396166157869408, 345.1465324932399873076, 348.3354227108952286379, 351.5234358145634796649, 354.7105963790346251997, 357.8969280552183801044, 361.0824536137919385758, 364.2671949863627303258, 367.4511733043118276415, 370.6344089354709198121, 373.8169215187742283183, 376.9987299970161740654, 380.1798526478359341892, 383.3603071130411575888, 386.5401104263749708533, 389.7192790398229327479, 392.8978288485497257227, 396.0757752145490528218, 399.2531329890843890468, 402.429916533992873612},
{68.04818346474532120283, 74.03967526028442591723, 79.16188325732047958084, 83.85039355752921311163, 88.26796669985123896783, 92.49678498001016180297, 96.58519571217465227164, 100.56445829029138438694, 104.45613332653570762762, 108.27581045228433297589, 112.0351755424767310683, 115.74324108365954004122, 119.40711990262181611472, 123.0325336942129815256, 126.6241595804643966897, 130.1858735543298537252, 133.7209259453434031978, 137.232070718075693046, 140.7216626007358382448, 144.1917312897622284301, 147.6440389937535420259, 151.0801256552267647054, 154.5013449154791256569, 157.9088930270026148694, 161.3038323243330364808, 164.6871104476105228569, 168.0595762159859811217, 171.4219928329428994015, 174.7750489478430570366, 178.1193679808447330747, 181.4555160303402572829, 184.7840086152614979711, 188.105316453401836594, 191.4198704373016776226, 194.7280659383541350515, 198.0302665455009245049, 201.3268073256520371567, 204.6179976776200952629, 207.9041238390429682826, 211.1854510958186016178, 214.462225735492072344, 217.7346767794305802883, 221.0030175231984170852, 224.2674469100678187584, 227.528150758890097263, 230.7853028644601919532, 234.0390659859221724497, 237.2895927365917789364, 240.5370263867414060859, 243.7815015893438404952, 247.0231450374556035397, 250.2620760607999143135, 253.4984071681511809053, 256.7322445413013802309, 259.9636884856821315268, 263.1928338421068691175, 266.4197703635704685029, 269.6445830605866401644, 272.8673525181460660464, 276.0881551870319666983, 279.3070636519272775007, 282.5241468784827414182, 285.7394704412827981245, 288.9530967344417561187, 292.1650851663826235785, 295.3754923401919438582, 298.5843722208032943333, 301.7917762901374145056, 304.9977536912162125011, 308.2023513621694193825, 311.4056141609649165228, 314.6075849816154533257, 317.8083048625444740839, 321.0078130877311046805, 324.2061472811981597199, 327.4033434953565749187, 330.5994362936742982918, 333.7944588280968178077, 336.9884429116096636661, 340.1814190862999616422, 343.373416687244044108, 346.564463902520907162, 349.7545878296266353852, 352.9438145285425344545, 356.1321690716893809292, 359.3196755909817093311, 362.506357322179223821, 365.6922366467170805008, 368.8773351311827898142, 372.0616735645947057408, 375.2452719936253825394, 378.4281497559023861472, 381.6103255115093523657, 384.7918172728011037724, 387.9726424326383965776, 391.1528177911402997616, 394.3323595810452509234, 397.5112834917654315515, 400.6896046922132095073, 403.8673378524729638087},
{69.08667234411945969724, 75.10513087299593754434, 80.24827425654414980727, 84.95445516963535069954, 89.38751791906904928621, 93.63021572202545078684, 97.73124522475312403458, 101.72210018745985456436, 105.62450863134446995376, 109.45418533485321647799, 113.22291301245767536188, 116.93978114157455432307, 120.61196510688623577452, 124.2452383508362793197, 127.8443214447873437791, 131.4131273278188819762, 134.954938088245851959, 138.4725352480430209652, 141.9682976444307560108, 145.4442762191997244863, 148.9022520234678320099, 152.3437818074112165092, 155.7702342824289128801, 159.1828192763173426317, 162.5826114042497442413, 165.9705694587652138784, 169.3475524226805333789, 172.7143327922031693932, 176.0716077386008982576, 179.4200085187526928096, 182.7601084562436708612, 186.0924297473630645778, 189.4174492947730888245, 192.7356037317099005859, 196.0472937684472370571, 199.3528879682761632134, 202.6527260408656315511, 205.9471217254033419016, 209.236365323499815266, 212.5207259318080810009, 215.8004534161611355983, 219.0757801623703781308, 222.3469226333592876166, 225.6140827577927770829, 228.8774491716196203219, 232.1371983308276035822, 235.3934955111030955592, 238.6464957078963512418, 241.8963444485470860415, 245.1431785265620200139, 248.3871266668088660263, 251.6283101292602859899, 254.8668432579545151492, 258.1028339810102680308, 261.3363842668204468148, 264.5675905409340875012, 267.7965440676039692027, 271.0233312995159368867, 274.2480341988148484044, 277.470730532192430819, 280.6914941424968810417, 283.9103951990565794467, 287.1275004286755618284, 290.3428733290519669565, 293.5565743661887588002, 296.768661157205384365, 299.9791886398169069908, 303.1882092296211814827, 306.3957729662227747815, 309.6019276491228291729, 312.8067189642153989073, 316.010190601651654159, 319.2123843657626012361, 322.4133402776676265844, 325.6130966711393739135, 328.8116902822444580059, 332.0091563332336493378, 335.2055286111138560128, 338.4008395412969797506, 341.5951202566870879738, 344.788400662536934655, 347.9807094973773349511, 351.1720743902979493647, 354.362521914835393947, 357.5520776397040262634, 360.7407661765860517056, 363.9286112251805640287, 367.1156356156956111878, 370.3018613489532143034, 373.4873096342643315949, 376.6720009252189322329, 379.8559549535255216147, 383.0391907610245449681, 386.2217267299910056393, 389.4035806118332918212, 392.5847695542875424469, 395.7653101271998380037, 398.9452183469820197268, 402.1245096998209710692, 405.303199163715693458},
{70.12476142569445869412, 76.16992633973481474641, 81.33381561851167668411, 86.05751650270140373777, 90.50594425483462744459, 94.76241614312439764944, 98.87597388006068399591, 102.87834269612460594992, 106.7914159336938219489, 110.63103194671629710311, 114.40906906733833503801, 118.13469278974169095007, 121.81514026424559515283, 125.4562360259826498508, 129.0627435469579795254, 132.6386122474611452874, 136.1871555768565426309, 139.7111822714468942503, 143.2130949794032659601, 146.6949656272856701543, 150.1585938796468947681, 153.6055530919338848463, 157.0372268660540422068, 160.4548384451089408383, 163.8594745818542267506, 167.2521050929053878681, 170.633599009295164701, 174.0047380158033787147, 177.3662277114095281874, 180.7187071043277762988, 184.0627566657703861614, 187.3989051987826582687, 190.7276357265153083497, 194.0493905640922760539, 197.3645757068642562414, 200.673564643173017537, 203.9767016802125561752, 207.2743048559869864994, 210.5666684978505496664, 213.8540654780047900399, 217.1367492081122853112, 220.4149554084734453263, 223.6889036816992873166, 226.9587989162620783884, 230.2248325415315943536, 233.4871836527608698566, 236.746020021855310324, 240.0014990075499842897, 243.2537683767572564318, 246.5029670472705332158, 249.7492257606710609269, 252.9926676931437930551, 256.2334090109329225895, 259.4715593763311543799, 262.7072224093772528288, 265.9404961098157078482, 269.1714732433354624741, 272.4002416956389955315, 275.6268847974881661777, 278.8514816235203090498, 282.0741072673197193436, 285.2948330949596416916, 288.5137269789928897566, 291.7308535146607888846, 294.9462742199064339114, 298.1600477206160302387, 301.3722299223685513308, 304.5828741698467094608, 307.7920313949492439557, 310.9997502545440117753, 314.2060772587117899905, 317.411056890250745308, 320.6147317161400445944, 323.8171424915970750135, 327.018328257305344474, 330.2183264303385848897, 333.4171728892602200286, 336.6149020538356065324, 339.8115469597568023059, 343.0071393287456138716, 346.2017096343699299854, 349.3952871638805159075, 352.5879000763502148911, 355.7795754573746097615, 358.9703393705723979568, 362.1602169061048152177, 365.3492322264162171593, 368.5374086093822260422, 371.7247684890375223272, 374.9113334940422734264, 378.0971244840342260719, 381.2821615840025371827, 384.466464216809385444, 387.6500511339762063432, 390.8329404448429502374, 394.0151496442010070658, 397.1966956384933103984, 400.3775947706685716631, 403.5578628437705521424, 406.7375151433377101283},
{71.16246099854829760596, 77.23407823516038206435, 82.4185282210675608029, 87.15960165273429028276, 91.62327230111553433289, 95.893414812799704805, 100.01941182431653487583, 104.03321722464711639011, 107.95688765023065895921, 111.80638350454515817764, 115.59367755082996023476, 119.32801035385192384419, 123.01668005982003136113, 126.66556166006057265202, 130.2794609937693781406, 133.8623635102670692959, 137.4176136329712254763, 140.9480469785489153729, 144.4560897158434917377, 147.9438345022821600876, 151.4130993925295954238, 154.8654741499033229025, 158.3023570915592651027, 161.7249847196812063861, 165.1344557848516138524, 168.5317510023292160065, 171.9177493384482232258, 175.2932415646378382299, 178.6589416153568881258, 182.0154961664984061614, 185.3634927608617855291, 188.7034667389942040384, 192.0359071813447112516, 195.3612620271689621307, 198.6799425040207954539, 201.9923269768159549162, 205.2987643057658705507, 208.5995767867738958293, 211.8950627352751964533, 215.1854987643123739294, 218.4711417993586746219, 221.7522308656345523215, 225.0289886781056651626, 228.3016230597626755574, 231.5703282099783893283, 234.835285841568113323, 238.0966662025273550907, 241.3546289961935121371, 244.6093242116998742776, 247.8608928750005048976, 251.1094677293942854443, 254.3551738533256466109, 257.5981292222556076807, 260.8384452205528922059, 264.0762271086289890919, 267.3115744499148054444, 270.5445815017348353324, 273.7753375736649021559, 277.0039273565509500192, 280.2304312250101942979, 283.4549255159247325959, 286.6774827851651678648, 289.8981720445425759684, 293.1170589807767365161, 296.3342061580830879685, 299.5496732058170799813, 302.7635169924696665445, 305.9757917871792016444, 309.1865494098108921857, 312.3958393705534446742, 315.6037089998920701943, 318.8102035697362515149, 322.0153664064084688261, 325.2192389961354203725, 328.4218610836252847825, 331.6232707642624886498, 334.8235045704045975662, 338.022597552223756075, 341.2205833534970484695, 344.4174942827157868153, 347.6133613798526578499, 350.8082144790975263442, 354.0020822678471898481, 357.1949923422112349887, 360.3869712592751163616, 363.5780445863424506532, 366.7682369473611000036, 369.9575720667217399447, 373.1460728106031171893, 376.3337612260249661614, 379.5206585777574498768, 382.706785383224912808, 385.8921614455315841547, 389.0768058847275629525, 392.2607371674248740018, 395.4439731348655358876, 398.6265310295363665798, 401.8084275204186117064, 404.9896787269543645763, 408.1703002418061093674},
{72.19978093323305522704, 78.29760247461114829326, 83.50243212779418880832, 88.26073379308877486646, 92.73952765123151209536, 97.0232392434871826481, 101.16158810634906765424, 105.18675405599625519613, 109.12095505385753410904, 112.98027207057003451987, 116.7767711476344630363, 120.51976700095686178531, 124.21661802445339384857, 127.87324904679930383575, 131.4945077555503277567, 135.0844151891114372122, 138.6463463682770431125, 142.183163464883531625, 145.6973158856641451317, 149.1909167714530827276, 152.6658023492653848026, 156.1235785977094508299, 159.5656583781227672077, 162.9932912992065316948, 166.4075879728146151852, 169.8095398902219440732, 173.2000358426153267016, 176.5798755883206670828, 179.9497813069569916096, 183.310407260128743417, 186.6623479876748122472, 190.0061452997099664309, 193.3422942719595565562, 196.6712484110891790113, 199.993424124898030025, 203.3092046072104310472, 206.6189432274661282395, 209.9229664991865793687, 213.2215766887877898074, 216.5150541159433701964, 219.8036591883575139533, 223.0876342069889830197, 226.3672049721659924153, 229.6425822164079525193, 232.9139628859349042761, 236.181531289650379688, 239.4454601317101981251, 242.705911441544051168, 245.9630374133029456244, 249.2169811651026243033, 252.4678774270715052115, 255.7158531660512118252, 258.961028153805497616, 262.2035154847422798359, 265.4434220484213244366, 268.6808489614884451026, 271.9158919631305993357, 275.1486417776722383574, 278.3791844475210393963, 281.6076016393117804141, 284.8339709257830873602, 288.058366045646744769, 291.2808571434678386718, 294.5015109913616352221, 297.7203911941259068157, 300.9375583792620865611, 304.1530703731923269552, 307.366982364849828133, 310.5793470577045947179, 313.7902148111842767294, 316.99963377235839345, 320.2076499986726830268, 323.4143075724473969114, 326.6196487077880518954, 329.8237138504985794423, 333.0265417715342008878, 336.2281696544840335823, 339.4286331775308066536, 342.6279665902966176825, 345.8262027859489379205, 349.0233733689096729161, 352.2195087184816541622, 355.414638048681163516, 358.6087894645416996244, 361.8019900151329398577, 364.9942657435195153709, 368.1856417338666081385, 371.376142155883325658, 374.56579030678015875, 377.7546086509034442017, 380.9426188571975152004, 384.1298418346340197142, 387.316297765737622708, 390.5020061383278949427, 393.6869857755885506327, 396.871254864567257931, 400.054830983201946643, 403.2377311259628196026, 406.4199717281930862717, 409.6015686892257328451},
{73.23673070487535165719, 79.36051434974713458814, 84.58554663121926742633, 89.36093522258007797655, 93.85473494917161366895, 98.15191594391060903035, 102.30253073211792687618, 106.338982402816994971, 110.28364832887322042936, 114.15272860743165768356, 117.95838143773283205687, 121.70999479298473490701, 125.41498658729880016443, 129.07933088478597926428, 132.7079167165675470094, 136.3048002819393826234, 139.873386832320977961, 143.4165647779636140105, 146.9368064879307093254, 150.4362453452166069829, 153.9167355367949406328, 157.3798990686446729744, 160.8271631792751850817, 164.2597904348842296902, 167.6789031751456340277, 171.0855035474485996835, 174.4804900597811785245, 177.8646713597264414121, 181.2387777836099267157, 184.6034710984290249372, 187.9593527679738109992, 191.3069710052850242576, 194.6468268204881695645, 197.9793792319533335973, 201.3050497766690093301, 204.6242264305028762612, 207.9372670290439177322, 211.2445022637804628115, 214.546238315567826497, 217.8427591769955307052, 221.1343287068571740312, 224.4211924530555449096, 227.7035792746314163398, 230.9817027889447704991, 234.2557626661721739066, 237.5259457900638015647, 240.7924273012091969959, 244.0553715367972504663, 247.3149328789468359339, 250.5712565220685763981, 253.824479168345478287, 257.074729659250127284, 260.322129550015610723, 263.5667936331191036794, 266.8088304160986913392, 270.0483425583869371299, 273.2854272712935311046, 276.5201766847912222479, 279.7526781843434125988, 282.9830147206492617861, 286.2112650948653371165, 289.4375042215853483394, 292.6618033716159113244, 295.8842303963719931954, 299.1048499355267891372, 302.3237236093839278342, 305.5409101972922402314, 308.7564658032924035308, 311.9704440100684812761, 315.1828960221739045936, 318.3938707994092102719, 321.6034151811465103866, 324.8115740023220376668, 328.0183902017521669168, 331.2239049233691638838, 334.4281576109197809316, 337.6311860966220235528, 340.8330266842323574036, 344.0337142269367895056, 347.2332822004441801726, 350.4317627716284196977, 353.629186863037379277, 356.825584213560504125, 360.0209834355232796747, 363.2154120684553223612, 366.4088966297593058134, 369.6014626624901367984, 372.7931347804375699007, 375.9839367106906414731, 379.1738913338487743193, 382.3630207220320319682, 385.5513461748316753744, 388.7388882533317970197, 391.9256668123232894257, 395.1117010308226678276, 398.2970094410002389928, 401.4816099556147258498, 404.6655198940446630229, 407.8487560070006195248, 411.0313344999965347752},
{74.2733194146759387877, 80.42282856177044436663, 85.66789029313914959543, 90.46022741043821866586, 94.96891793759650990842, 99.27947046903390562433, 103.442266715820617699, 107.48993045910000393928, 111.44499662275857233764, 115.32378302974292857777, 119.13853894746215808511, 122.89872473713366276428, 126.61181712537673580391, 130.28383882607461476221, 133.9197197226013360188, 137.5235507582480851622, 141.0987670578558965574, 144.6482829614643160639, 148.1745935318687473106, 151.6798521589713384608, 155.165930782498387402, 158.6344672523836141513, 162.0869030212252596683, 165.524513467130398778, 168.9484325271521671643, 172.3596728875385695596, 175.7591426673587005149, 179.1476593078705165447, 182.5259612154708631052, 185.8947175838412886515, 189.2545367290612773181, 192.6059732017214066229, 195.9495338865947069575, 199.2856832590483939279, 202.6148479350912441885, 205.9374206265576319985, 209.2537635928077451788, 212.5642116642687202266, 215.8690749002474151276, 219.1686409330259899873, 222.4631770417823888445, 225.7529319929561228238, 229.0381376779931750546, 232.3190105747088472107, 235.5957530546128115603, 238.8685545552956075125, 242.1375926342604885853, 245.4030339183032021677, 248.6650349606181783943, 251.9237430161807791719, 255.1792967445715221439, 258.4318268482286890432, 261.6814566531070492142, 264.9283026378551534841, 268.172474916879177749, 271.4140776820189181938, 274.6532096070057311353, 277.8899642183900284549, 281.1244302362065631421, 284.3566918872800906632, 287.5868291937544313107, 290.8149182391480380806, 294.0410314139934351558, 297.2652376429016995025, 300.4876025947025674876, 303.7081888771423928003, 306.9270562174731802308, 310.1442616301338006988, 313.3598595726071313694, 316.5739020904324294465, 319.7864389522591574536, 322.9975177757453622684, 326.2071841450293820772, 329.4154817204370823359, 332.6224523410271045492, 335.8281361205229655461, 339.0325715371325839181, 342.2357955177123335495, 345.4378435166935045196, 348.6387495901536257347, 351.8385464653830635155, 355.0372656062682962972, 358.2349372747869598598, 361.4315905888858790446, 364.627253576991601718, 367.8219532293832079016, 371.0157155466391851903, 374.2085655853537661935, 377.4005275013031591228, 380.5916245902284300068, 383.7818793263892903007, 386.9713133990315959511, 390.1599477469008739337, 393.3478025909245708135, 396.5348974651768850227, 399.7212512462319284347, 402.9068821810034987667, 406.091807913162874108, 409.276045508219711927, 412.4596114773452998261},
{75.30955580994140931457, 81.48455925242124223253, 86.74948098228896733112, 91.55863103835298488775, 96.08209950278301903266, 100.40592746687984515574, 104.58082212784074727097, 108.63962544870434503444, 112.60502809485312647498, 116.49346425259782401479, 120.31727319760985961552, 124.08598683336152557462, 127.80714001031554640769, 131.48680352206734924744, 135.1299476258856043881, 138.7406976030258994253, 142.3225181037369645803, 145.8783490970493298465, 149.4107080776041247987, 152.9217682127439524927, 156.4134189927512741959, 159.8873139324543823521, 163.3449085392581147844, 166.7874908609187432451, 170.2162063043486702806, 173.6320779799693162882, 177.0360235144771554291, 180.4288690492253997715, 183.8113609758194321423, 187.1841758374871196245, 190.547928732329673955, 193.9031804843498197575, 197.2504437943165985106, 200.5901885408666748067, 203.9228463697321771297, 207.2488146834140675554, 210.5684601233573086855, 213.8821216205157099561, 217.1901130772083275167, 220.492725732674803557, 223.7902302562065468595, 227.0828786047583680003, 230.3709056762166992084, 233.6545307847706517924, 236.9339589809084443047, 240.2093822352922143807, 243.4809805030282327741, 246.7489226825507167519, 250.0133674813984582911, 253.2744641995219642836, 256.5323534393642011723, 259.7871677507691934925, 263.0390322177559989973, 266.2880649933233493548, 269.5343777876997361236, 272.7780763148060977167, 276.0192607011378794461, 279.2580258607870573919, 282.4944618399018352855, 285.7286541335129953536, 288.9606839773336117512, 292.1906286168565212424, 295.4185615558260893845, 298.6445527859427415453, 301.8686689994664710156, 305.090973786215697887, 308.3115278163075284901, 311.5303890098521650927, 314.7476126946957973575, 317.9632517532009247309, 321.1773567589591178984, 324.3899761042473473217, 327.6011561189639901887, 330.8109411817134329374, 334.0193738236479082149, 337.2264948256210515047, 340.4323433091589419452, 343.636956821710498159, 346.8403714165995015669, 350.0426217280647493635, 353.2437410417424856511, 356.4437613609159593318, 359.6427134688303907687, 362.8406269873475124241, 366.0375304321919304866, 369.2334512650216120377, 372.4284159425366375371, 375.6224499628237951208, 378.8155779091194743976, 382.0078234911595030028, 385.1992095842719339968, 388.3897582663572243133, 391.5794908528896436292, 394.7684279300640293285, 397.9565893862030759651, 401.1439944415321442403, 404.3306616764210297382, 407.5166090581851866791, 410.70185396653250376, 413.886413217735829844},
{76.34544830276801974378, 82.54572003292788906492, 87.83033590956901456932, 92.65616603983552513716, 97.1943017167447888915, 101.5313107224514361338, 105.71822213977255845106, 109.78809367096371829575, 113.76376996214697403681, 117.66180023724499302671, 121.49461274873474220166, 125.27181011917381119109, 129.00098465246721758861, 132.68825466685229632323, 136.33863032758817870085, 139.9562708583165092661, 143.5446700955279282634, 147.1067933439911946549, 150.6451802747802475793, 154.162023608795150314, 157.6592301895182145805, 161.1384690218246340907, 164.6012095123231220024, 168.0487522393833037599, 171.4822539550901376821, 174.9027480818490691978, 178.3111616527339610656, 181.7083294175630130684, 185.0950056700147156586, 188.4718742272582957065, 191.8395569005110711359, 195.198620724261823357, 198.5495841577028378901, 201.8929224299737218064, 205.2290721680899276113, 208.5584354206831539593, 211.8813831702784957812, 215.198258410552561137, 218.5093788519398658807, 221.8150393083862796885, 225.1155138094571386193, 228.411057474985362433, 231.7019081836750322554, 234.9882880623115521155, 238.2704048182769882647, 241.5484529347754630013, 244.8226147454170765479, 248.0930614024927260464, 251.3599537513185164598, 254.6234431213744037446, 257.8836720435563719738, 261.140774901663381545, 264.3948785252156509536, 267.6461027298217297419, 270.894560810555353033, 274.1403599931502633617, 277.3836018472562847303, 280.6243826655098035509, 283.8627938117454687068, 287.098922041304154229, 290.3328497960672783388, 293.5646554765628956185, 296.7944136932390326026, 300.0221954987798156435, 303.2480686031460375497, 306.4720975728505068071, 309.694344015826893995, 312.914866753116324023, 316.133721978476501412, 319.3509634069118385032, 322.5666424130282760043, 325.7808081600318503205, 328.9935077201143656055, 332.2047861869017211047, 335.4146867805796121628, 338.6232509462566685885, 341.8305184460759194846, 345.0365274455411682671, 348.2413145944848893377, 351.4449151030681475561, 354.6473628131703783772, 357.8486902654972843986, 361.0489287627082798923, 364.2481084288405627447, 367.4462582652847597581, 370.6434062035469516761, 373.8395791550135388279, 377.0348030579186791948, 380.2291029216987595961, 383.4225028689044062446, 386.6150261748277768927, 389.8066953049911902094, 392.9975319506324377648, 396.187557062312299314, 399.3767908817607615406, 402.565252972070150821, 405.7529622463357660386, 408.9399369948375786617, 412.1261949108501006716, 415.3117531151615581879},
{77.38100498748636368873, 83.60632401107309350367, 88.91047166101760979018, 93.75285163710216806976, 98.30554587674225633191, 102.6556431989714154841, 106.85449106673592846018, 110.93536054358707432852, 114.92124854239355509054, 118.82881803412545592842, 122.67058524390735966431, 126.45622271189453363712, 130.19337954257594420224, 133.88822103816784128155, 137.54579681799360138635, 141.1702996625628145464, 144.7652522639646425679, 148.3336449767259307675, 151.8780393991857237545, 155.4006475873107506227, 158.9033935451046762148, 162.3879615967168834308, 165.8558348959200241124, 169.3083264157862757773, 172.7466041316353485173, 176.1717116680914772305, 179.5845853654984722165, 182.9860684924066536201, 186.3769231631154161099, 189.7578403946258934636, 193.129448643696372147, 196.492321093560621151, 199.8469819053178830196, 203.1939116067868325565, 206.5335517586678026446, 209.8663090119391009228, 213.1925586498729806103, 216.5126476916652046734, 219.8268976215053955412, 223.1356067962736411648, 226.4390525763977076781, 229.7374932173335326078, 233.0311695533208807326, 236.3203065002676671328, 239.6051144006354562201, 242.8857902298810211912, 246.1625186812322817538, 249.4354731432437588579, 252.704816582608496198, 255.9707023430369598792, 259.2332748695974858288, 262.4926703667056710765, 265.7490173969175798992, 269.0024374267957442057, 272.2530453253545858061, 275.5009498199339667673, 278.7462539137802120146, 281.989055269119918109, 285.229446559082101823, 288.4675157914494707416, 291.703346606892004316, 294.9370185540490232761, 298.1686073435739148961, 301.3981850830339308236, 304.6258204943619471436, 307.8515791153843279098, 311.0755234867961152179, 314.2977133258191516444, 317.518205687658249359, 320.7370551157632795978, 323.9543137818094468082, 327.1700316162226314396, 330.3842564300003175426, 333.5970340285102084701, 336.8084083178872548109, 340.0184214045946692294, 343.2271136886648801067, 346.4345239510916639457, 349.6406894358043550119, 352.8456459266185845656, 356.0494278195250330228, 359.2520681906478175403, 362.4535988601770589655, 365.6540504525555875071, 368.8534524531774003982, 372.0518332618351505345, 375.2492202431354211312, 378.4456397740836485791, 381.64111728902513414, 384.8356773221144923726, 388.0293435474729928512, 391.222138817181447916, 394.4140851972454808162, 397.6052040016600841621, 400.7955158246912659277, 403.9850405714842054736, 407.1737974870996387229, 410.3618051840730998496, 413.5490816685851126045, 416.7356443653243989634},
{78.41623365696529876333, 84.66638381652306097885, 89.98990422870345184947, 94.84870637566780260386, 99.41585254237627111706, 103.77894707663672591025, 107.98965240717808540294, 112.08145064304655964092, 116.07748929473195041014, 119.99454382345748006448, 123.84521744904657941783, 127.63925184859054030676, 131.38435229116299642892, 135.08673053614999601232, 138.75147521453815620598, 142.3828122878735012489, 145.9842929814121600639, 149.5589324204718613137, 153.1093138875160758211, 156.6376685602955335746, 160.1459374151787699545, 163.6358199287593828388, 167.1088128533843703694, 170.5662414239469387227, 174.0092847197310123179, 177.4389964601689008277, 180.8563221958500787612, 184.2621136261708907799, 187.6571406062395476074, 191.0421012802384963873, 194.4176306841912614657, 197.7843080894942818575, 201.1426633036718359672, 204.4931821023219052295, 207.8363109330574404216, 211.1724610061582516802, 214.5020118659720845277, 217.8253145206021153072, 221.1426941941625396378, 224.4544527551698633005, 227.7608708659271143913, 231.0622098906376925363, 234.3587135941343460915, 237.6506096582767431015, 240.938111039061973913, 244.2214171841510734306, 247.5007151277181593656, 250.7761804771787316743, 254.0479783043711475277, 257.3162639520866117754, 260.5811837654166049871, 263.8428757561704982724, 267.1014702075758231216, 270.3570902255810681728, 273.6098522423127098659, 276.8598664765752079285, 280.1072373557089276511, 283.3520639026230683133, 286.5944400913875407272, 289.83445517438999959, 293.0721939837340309203, 296.3077372092651794826, 299.5411616553574521376, 302.7725404783693754712, 306.0019434064815543613, 309.2294369434535018, 312.4550845576833190951, 315.6789468578170512798, 318.9010817560330336745, 322.1215446200183930588, 325.3403884145584388833, 328.5576638335735634633, 331.7734194233612392848, 334.9877016977316906611, 338.2005552456638976167, 341.4120228320529517468, 344.6221454920697183575, 347.8309626196086457336, 351.0385120502588561902, 354.2448301391968757975, 357.4499518343660886588, 360.6539107452778654823, 363.8567392077419863633, 367.0584683448091635204, 370.2591281241859136233, 373.4587474123615027374, 376.6573540256679866717, 379.8549747784773148645, 383.0516355287238957967, 386.2473612209267925314, 389.4421759268726998964, 392.6361028841089351582, 395.8291645323847488471, 399.0213825481692393796, 402.2127778773649514032, 405.4033707663277788548, 408.5931807912960125241, 411.7822268863242080332, 414.9705273698109491036, 418.1580999697034935767},
{79.4511418178645953958, 85.72591162455351820881, 91.06864903969508863689, 95.94374815681953480246, 100.52524157044295805008, 104.9012437890679194652, 109.12372888034133211477, 113.22638774262996718599, 117.23251685799178476988, 121.15900295353689100635, 125.01853529101512167787, 128.82092392380608995046, 132.57392966577865229949, 136.28381022000737296328, 139.95569279783547245683, 143.5938361753445077896, 147.2018197964415752049, 150.7826832850329567377, 154.3390313703841605221, 157.8731141437790194741, 161.386889370166956619, 164.8820715155714472114, 168.3601707856663071937, 171.8225245472211300689, 175.2703228668018533591, 178.7046294535251978702, 182.1263989732274946776, 185.5364914700624913983, 188.9356844617331470305, 192.3246831483755943285, 195.7041290802717287551, 199.0746075575311749919, 202.4366539796337769349, 205.7907593199627301984, 209.1373748670820880801, 212.4769163482542585033, 215.8097675298815740106, 219.1362833729462125155, 222.4567928081793736011, 225.7716011849050260278, 229.080992438734554488, 232.3852310161198606185, 235.684563587881278342, 238.9792205789613777097, 242.2694175376187961052, 245.5553563639116339439, 248.8372264145037670415, 252.1152054984607015652, 255.3894607767048927203, 258.6601495761096817307, 261.9274201277742352484, 265.1914122377968054412, 268.4522578978156842193, 271.7100818416880037584, 274.9650020539026266488, 278.2171302346553988443, 281.4665722259369154739, 284.7134284024812608849, 287.9577940309877124422, 291.1997596006467302174, 294.4394111276687692226, 297.6768304362228585092, 300.9120954179358286339, 304.1452802718777265304, 307.3764557267602399736, 310.6056892468993669357, 313.8330452233381164009, 317.0585851513871497145, 320.2823677957187576804, 323.5044493440405153899, 326.7248835502777200645, 329.943721868106873535, 333.1610135756047852486, 336.3768058917082700782, 339.5911440851169622254, 342.804071576215645665, 346.015630032541998863, 349.2258594582801413939, 352.4347982782193036364, 355.6424834165798353265, 358.8489503710751994186, 362.0542332825481895959, 365.258365000492031726, 368.4613771447419885633, 371.663300163600323408, 374.8641633886367618159, 378.0639950863877160222, 381.2628225071603222567, 384.460671931131624146, 387.6575687119188710025, 390.8535373177837584638, 394.0486013706214048071, 397.2427836828738255342, 400.4361062924975484007, 403.628590496105717377, 406.8202568803964919166, 410.0111253519716897583, 413.2012151656423863505, 416.3905449513115169106, 419.5791327395173784918},
{80.48573670491761333778, 86.78491917829424551201, 92.14672098325107135108, 97.03799426812678063925, 101.63373214771218792614, 106.02255405761832132588, 110.25674246156154367782, 114.37019484832002332436, 118.38635508683983693615, 122.32221997690762768206, 126.19056389362399542667, 130.00126452525235559871, 133.76213762626035896128, 137.47948634275734455247, 141.1584760458206630082, 144.80339796855778167624, 148.4178594666435557891, 152.0049243968983048803, 155.5672187036857290601, 159.1070111884346195335, 162.6262762251202996056, 166.1267431098752322586, 169.6099353596903132601, 173.0772023461146256053, 176.5297450088259598958, 179.9686369437234831272, 183.3948418388610545758, 186.8092279988107053394, 190.2125805272130240547, 193.6056116103180452004, 196.9889692488980503719, 200.3632447134336652798, 203.7289789418816028214, 207.086668056303519877, 210.4367681410483843229, 213.77969939875560999, 217.1158497795012857524, 220.4455781616927412137, 223.7692171498864779968, 227.0870755438481772335, 230.3994405243465638077, 233.7065795939564064501, 237.0087423052152180596, 240.3061618035800501993, 243.5990562085663487601, 246.8876298530631504464, 250.1720743979832468074, 253.4525698370237326651, 256.7292854043016404842, 260.00238039592664325, 263.2720049151258055568, 266.5383005493015072289, 269.8014009863481446289, 273.0614325766474461025, 276.3185148463826527199, 279.5727609671389099565, 282.8242781861747900864, 286.0731682212444152846, 289.3195276234098836573, 292.5634481109001377137, 295.8050168767370799409, 299.0443168725558979767, 302.2814270707885092466, 305.5164227071519289172, 308.7493755051830819778, 311.9803538843845997324, 315.2094231533894497866, 318.4366456894132614118, 321.6620811051397014006, 324.885786404074312369, 328.1078161253041898182, 331.3282224785133121701, 334.5470554700250052255, 337.7643630205728372453, 340.9801910754382606728, 344.1945837075367187588, 347.407583213983000607, 350.6192302066207261774, 353.8295636969594203963, 357.0386211759252064183, 360.2464386887972840122, 363.4530509056716822168, 366.6584911877659521089, 369.8627916498532004211, 373.0659832190908961412, 376.2680956904889778589, 379.4691577792427430858, 382.6691971701386283341, 385.8682405642251266693, 389.0663137229265916423, 392.2634415097644123759, 395.4596479298378973258, 398.654956167206069195, 401.8493886203013567993, 405.0429669354967870711, 408.2357120389396560766, 411.4276441667567237076, 414.6187828937286711434, 417.8091471605248278303, 420.9987552995829655473},
{81.52002529431781292006, 87.84341780960309620241, 93.22413443636222041278, 98.13146141213034923112, 102.74134282177827475544, 107.14289792369405063255, 111.38871441554835033178, 115.51289423264992800209, 119.539027085914725167, 123.48421868454512384422, 127.36132761168406187134, 131.18029846758536425795, 134.94900135812569574841, 138.67378438414691258102, 142.35985066613144479725, 146.01152354537073318511, 149.6324379897870056929, 153.2256818297413007076, 156.7939019984190079056, 160.3393858087064778579, 163.8641240681411849788, 167.3698607472196613618, 170.8581325353775187414, 174.3303006846081850211, 177.7875768959694490628, 181.2310445513993730985, 184.6616762700551561961, 188.0803485342908368764, 191.4878539585444184108, 194.884911646693559567, 198.2721759874424250525, 201.6502441643826361086, 205.0196626014384462172, 208.3809325211123836803, 211.7345147591521046392, 215.0808339526688326467, 218.4202821976608745843, 221.7532222550725039535, 225.07999037100239518, 228.4008987657495192234, 231.7162378375001753976, 235.0262781191961313887, 238.3312720211540471118, 241.631455387075669143, 244.927048886996671108, 248.2182592673115014586, 251.5052804751566983887, 254.7882946720356244721, 258.0674731495430066536, 261.3429771583331319035, 264.6149586600184303798, 267.8835610104426337517, 271.1489195817096898272, 274.4111623294373739643, 277.6704103109193415963, 280.926778159201582502, 284.1803745174925588085, 287.4313024378171416364, 290.6796597473814360575, 293.9255393857291598647, 297.1690297154323872991, 300.4102148087634032901, 303.6491747125343947154, 306.8859856930628563894, 310.1207204630187587295, 313.3534483917311663228, 316.5842357003740776429, 319.8131456433111723768, 323.0402386767546648978, 326.265572615782639886, 329.4892027806604209696, 332.7111821333232512508, 335.9315614047985924481, 339.1503892142755874319, 342.3677121804657298016, 345.5835750258417151298, 348.7980206742900868644, 352.0110903426670006514, 355.222823626704656736, 358.4332585816782011221, 361.6424317982087408319, 364.850378473547176245, 368.0571324786554878272, 371.2627264213766279819, 374.4671917059609975527, 377.6705585891963966017, 380.8728562333691224476, 384.0741127562663594134, 387.2743552784139992689, 390.473609967729401683, 393.6719020817552185146, 396.869256007628146707, 400.0656952999252364145, 403.2612427165200691583, 406.4559202525716503058, 409.6497491727601545959, 412.8427500418756540368, 416.034942753858582306, 419.2263465593838929167, 422.4169800920745993839},
{82.55401431627674743597, 88.90141845867101457864, 94.30090328776576313885, 99.22416573334110734894, 103.84809153011933346988, 108.26229477922366480101, 112.51966532778598239513, 116.65450746667248430735, 120.69055524208446960847, 124.64502213818397173828, 128.53085006323230257337, 132.35804982439572065021, 136.13454530421904101371, 139.86672908187263075248, 143.55984162683598899896, 147.21823804810155764526, 150.8455806334235177616, 154.4449809334147961882, 158.0191066490502393455, 161.5702634105317754401, 165.1004582874542551941, 168.6114497723963728788, 172.1047875914067604749, 175.581844755266849486, 179.0438436170496234365, 182.4918772460856338305, 185.9269271033836571208, 189.3498777681009460022, 192.7615292918105847959, 196.1626076288515406217, 199.5537734944820389213, 202.935629929202185073, 206.308728791342705751, 209.6735763564605815108, 213.0306381680806144041, 216.3803432575681281431, 219.7230878297105609275, 223.0592384936575310131, 226.3891351052678755161, 229.7130932759166885895, 233.0314065938744645962, 236.3443485970600605371, 239.6521745299607405209, 242.9551229125497140003, 246.2534169449130827899, 249.5472657678650233908, 252.8368655969560940961, 256.1224007448639331893, 259.4040445451173506147, 262.6819601883785962153, 265.956301481041467278, 269.2272135346517831057, 272.4948333935863451853, 275.7592906075078623194, 279.0207077543225769266, 282.2792009186847224855, 285.5348801305010593202, 288.7878497673758941743, 292.0382089244907437504, 295.2860517550235497143, 298.531467783872007882, 301.7745421971473110054, 305.0153561096426418759, 308.2539868122501821073, 311.4905080010970414535, 314.7249899899907936251, 317.9574999076061721124, 321.1881018807033119346, 324.4168572045424656063, 327.6438245015484299686, 330.8690598691783151467, 334.0926170178573132779, 337.3145473997675206276, 340.5349003292035348023, 343.7537230951445327828, 346.9710610666350007335, 350.1869577915145029999, 353.401455088990205772, 356.6145931365037491678, 359.8264105513059963879, 363.0369444671187453805, 366.2462306062312836602, 369.4543033473513617082, 372.6611957895044549186, 375.8669398122518125998, 379.0715661324765192559, 382.275104357967408608, 385.4775830380129879897, 388.6790297112013835518, 391.8794709506075567048, 395.0789324065355367793, 398.2774388469710453023, 401.4750141958885472348, 404.6716815695463586265, 407.8674633108938827499, 411.0623810222062608758, 414.2564555960536401191, 417.4497072447048167613, 420.6421555280581527853, 423.8338193801863351634},
{83.58771026681487520753, 89.95893169245045718096, 95.3770409605412368137, 100.31612284366792664514, 104.95399562749017263741, 109.38076339540469916756, 113.64961513418238434563, 117.79505545016932553091, 121.84096125495120646954, 125.80465270091118616275, 129.69915416005065119463, 133.53454195852417940798, 137.31879319472200256507, 141.0583444612055198146, 144.758473185609275384, 148.42356591220804466917, 152.0573119630311279051, 155.6628463615316926541, 159.2428573605107356472, 162.799668717740237275, 166.3353035971996110908, 169.8515348646221891469, 173.3499251497854578963, 176.8318591032012982407, 180.2985696228912868513, 183.7511593689698968005, 187.1906185568570226927, 190.6178397831477396844, 194.0336304643277558151, 197.4387233393182433914, 200.8337853897061557631, 204.2194254577308336959, 207.5962007854958853454, 210.9646226550597134585, 214.3251612748522476573, 217.6782500309498195576, 221.0242892004034975327, 224.363649206784141139, 227.696673484422281293, 231.023681006756991256, 234.3449685252109119154, 237.6608125576520146333, 240.9714711594559936842, 244.2771855051885140426, 247.5781813047814494028, 250.874670074621827721, 254.1668502810794368902, 257.4549083715674805289, 260.7390197061788635177, 264.0193494012029125553, 267.2960530943503254613, 270.5692776402545124974, 273.8391617437397636357, 277.1058365374217094631, 280.3694261094093153733, 283.630047986190274125, 286.8878135751866222987, 290.1428285709509312509, 293.3951933285239943209, 296.645003207082882948, 299.8923488866654374664, 303.1373166604568223597, 306.3799887048598916021, 309.6204433293388354787, 312.8587552078207075497, 316.0949955932583659646, 319.329232516798031331, 322.5615309728524244086, 325.7919530912540307267, 329.0205582975504884732, 332.2474034624037193601, 335.4725430409647550225, 338.6960292030159828901, 341.9179119546006401757, 345.1382392517948597288, 348.3570571072195763675, 351.5744096898374017181, 354.7903394185325253643, 358.0048870499292347876, 361.218091760866268982, 364.4299912259094928726, 367.6406216902539154529, 370.8500180383375325561, 374.0582138584635533364, 377.2652415037040001404, 380.4711321493362168774, 383.6759158470442700427, 386.8796215760993911954, 390.0822772917173223719, 393.2839099707755370709, 396.4845456550596853938, 399.6842094921961330811, 402.882925774416023379, 406.0807179752857917806, 409.2776087845294203911, 412.4736201410588533348, 415.6687732643208374012, 418.863088684060940035, 422.0565862685985730301, 425.2492852517004635771},
{84.62111941884149788736, 91.01596772199215479304, 96.45256043338879003612, 101.40734784638483284619, 106.05907191176376182271, 110.49832194984444943677, 114.77858314908417014522, 118.93455844021685729383, 122.99026616571773844805, 126.96313206613713995094, 130.86626213658637297159, 134.70979755080864334676, 138.50176807562957885086, 142.2486538631192152915, 145.9557689174524130231, 149.62753089355055799866, 153.2676558687843124933, 156.8793020977142411445, 160.4651781739051213727, 164.0276257972070266778, 167.5686840620210866778, 171.0901400615576499032, 174.5935691992967125116, 178.080367648944650848, 181.5517787486367141495, 185.0089146546431687466, 188.452774251116303459, 191.8842580742942023838, 195.3041808347556186292, 198.7132819913800869545, 202.1122347329828723524, 205.5016536493828009876, 208.882101316729801666, 212.2540939778465102053, 215.6181064639304727176, 218.9745764768877518088, 222.3239083301042681131, 225.666476228326352425, 229.0026271535536303633, 232.3326834127156693263, 235.656944893851323421, 238.9756910701074297343, 242.2891827847889828924, 245.5976638456667713968, 248.9013624525770578029, 252.2004924788703771909, 255.4952546243551383948, 258.7858374549343823053, 262.0724183420688414288, 265.3551643134502409757, 268.6342328247819868231, 271.9097724612963527106, 275.1819235765523097934, 278.4508188751269180171, 281.7165839450115463978, 284.979337744832099145, 288.2391930504132743718, 291.496256864686814443, 294.750630794491130096, 298.0024113974148662088, 301.2516905014917356706, 304.4985555002513635305, 307.7430896253650995953, 310.985372198891799721, 314.2254788669222081575, 317.4634818162381784895, 320.6994499754414527435, 323.9334492018634200927, 327.1655424554399127471, 330.3957899606216965167, 333.6242493572901743973, 336.8509758415574687435, 340.0760222972492014836, 343.2994394187958395711, 346.5212758261934448547, 349.7415781726362172853, 352.9603912453706075128, 356.1777580602733507115, 359.393719950612968463, 362.6083166504155989998, 365.8215863728210070481, 369.0335658837829038291, 372.2442905714389316838, 375.4537945114495320325, 378.6621105285811502845, 381.8692702547875975175, 385.0753041840236734773, 388.2802417240071692944, 391.4841112451289424663, 394.6869401266957404038, 397.8887548006767075388, 401.0895807931119241045, 404.2894427633297842722, 407.4883645411094304818, 410.6863691619147325642, 413.883478900317356385, 417.0797153017182368358, 420.2750992124691906087, 423.4696508084894180566, 426.6633896224651988746},
{85.65424783257523233943, 92.07253641876778260666, 97.52747426068212632799, 102.49785535873807863838, 107.16333664832686167756, 111.61498805220280416961, 115.90658809176590439486, 120.07303607821652599991, 124.13849038452185840567, 128.12048128504779769019, 132.0321955773748626489, 135.88383862736039827329, 139.68349233578660361926, 143.43767997101252619598, 147.15175174104267621291, 150.83015609432349678293, 154.47663559103112855038, 158.0943714805895921378, 161.6860924910050200333, 165.2541580828300849707, 168.8006231205175541619, 172.327288782226571009, 175.8357431178847036612, 179.3273937103040104159, 182.8034942350659041099, 186.2651662518932094944, 189.7134172297055704088, 193.1491555681182555884, 196.5732032023503611192, 199.9863062478400430557, 203.3891440426284279728, 206.7823368709402903903, 210.1664525941322607627, 213.5420123708525639683, 216.9094956136485070088, 220.2693443020237112607, 223.6219667503560619608, 226.9677409118507395144, 230.307017285851997397, 233.6401214846386055911, 236.9673565067204655673, 240.2890047562063927944, 243.6053298416910954996, 246.9165781830520871055, 250.2229804503498342036, 253.5247528555250772169, 256.822098314657442779, 260.1152074960865366848, 263.4042597676182392378, 266.6894240542784136396, 269.9708596165797647983, 273.2487167579912409185, 276.5231374692072393458, 279.7942560158764648772, 283.0621994756432710025, 286.3270882296575602266, 289.5890364131061037237, 292.8481523287945194991, 296.1045388273534578369, 299.3582936572449938446, 302.6095097873975778303, 305.8582757049931854422, 309.1046756906626466965, 312.348790073109510108, 315.5906954649749517177, 318.8304649815725284654, 322.06816844395888464, 325.303872567662174635, 328.5376411382616643171, 331.7695351748977333836, 334.9996130826896079918, 338.2279307949471222174, 341.4545419059813502261, 344.6794977952459480754, 347.9028477434755185851, 351.1246390414284124426, 354.3449170917883581934, 357.5637255047315197725, 360.7811061876224393538, 363.9970994292633313113, 367.2117439790859058742, 370.425077121642926792, 373.6371347467276997689, 376.8479514154233411549, 380.0575604223597177805, 383.2659938544341378107, 386.4732826462319947467, 389.679456632365431439, 392.884544596931528137, 396.0885743202763763799, 399.2915726232375432752, 402.4935654090247369932, 405.6945777028868453567, 408.8946336897028376268, 412.0937567496242073383, 415.2919694918876124971, 418.4892937869080676762, 421.6857507967553965867, 424.8813610041096058731, 428.0761442397843390827},
{86.68710136535211882019, 93.12864733004980166697, 98.60179459138103204742, 103.58765953328607167805, 108.26680559312719604455, 112.73077876843670329858, 117.03364811149354807185, 121.21050741548880580467, 125.28565371633639582381, 129.27672079263415517496, 133.19697544305830652172, 137.05668658546027891176, 140.86398773257222695935, 144.62544483611110028508, 148.34644394379583529382, 152.03146398773368706369, 155.6842737445528327178, 159.3080772276038609297, 162.9056230975974433939, 166.4792883983982423851, 170.0311436076207379274, 173.5630038488973521504, 177.0764696940364456856, 180.5729600232422533005, 184.0537387489801614374, 187.5199367435934638608, 190.972569978471233086, 194.4125546418287075285, 197.840719825404479684, 201.2578182389893164443, 204.6645353129193960486, 208.0614969736153169529, 211.4492763196680128296, 214.8283993813941709024, 218.1993501119779906764, 221.5625747309249861592, 224.9184855188372285098, 228.2674641451820904827, 231.6098645967943395174, 234.9460157635863069527, 238.2762237287790884046, 241.600773803475455136, 244.9199323392361325677, 248.2339483472329840417, 251.543054948329437655, 254.8474706749174992499, 258.1474006423926405422, 261.4430376056694844472, 264.7345629140496249795, 268.0221473759812198459, 271.3059520437439445991, 274.5861289268083283877, 277.8628216415192750331, 281.1361660038100364028, 284.4062905708405794019, 287.6733171367529195292, 290.9373611871287561286, 294.1985323162076083968, 297.4569346104648810326, 300.7126670017490344304, 303.96582359282700378, 307.2164939578802012625, 310.464763420223914465, 313.7107133092856444541, 316.9544211986686182184, 320.1959611269416987549, 323.4354038026330655283, 326.6728167947596566883, 329.9082647100951367841, 333.1418093582640832171, 336.3735099056474459664, 339.6034230189926339543, 342.8316029995395209478, 346.058101908400116318, 349.282969683863628552, 352.5062542512393041262, 355.7280016257960007, 358.9482560093092939255, 362.1670598806834441438, 365.3844540810762551918, 368.6004778939192952669, 371.815169120193724761, 375.0285641492927395213, 378.2406980257750815483, 381.4516045122899191818, 384.6613161489314125597, 387.8698643092612417039, 391.0772792532190918618, 394.2835901771243924145, 397.4888252609573388587, 400.6930117130932554331, 403.8961758126515565814, 407.098342949608829059, 410.2995376628147847741, 413.4997836760399391158, 416.699103932174771164, 419.8975206256917493008, 423.0950552334738939477, 426.2917285441064402951, 429.4875606857216054823},
{87.71968568086455775301, 94.1843096934139083487, 99.675533186881195624, 104.67677407805749112413, 109.36949401446172203123, 113.84571064373785986845, 118.15978081125471513134, 122.34699093752283527657, 126.43177538552563419729, 130.43187043238772799097, 134.36062209508685206482, 138.22836221815893477555, 142.04327541631394334448, 145.81196990162693420167, 149.53986720571668749265, 153.231476441498755123, 156.89059234167618794363, 160.5204414577221092601, 164.1237921857525277476, 167.7030389794120534263, 171.2602677759588743787, 174.7973075079828355032, 178.3157711472142593963, 181.817088761842057601, 185.3025344026987292891, 188.773248165735088207, 192.2302544441336894941, 195.6744771413819382603, 199.1067524389148762232, 202.5278395798340107499, 205.9384300308856993991, 209.3391553094173427159, 212.730593704129655522, 216.1132760736154309987, 219.4876908716734108762, 222.8542885208393729981, 226.2134852337361946619, 229.5656663644075856299, 232.9111893577872669152, 236.2503863541245419889, 239.5835664959716206877, 242.911017977801466806, 246.2330098721292351945, 249.5497937608917325247, 252.861605196590633589, 256.1686650141627816549, 259.4711805115747986789, 262.7693465146455867027, 266.0633463394957351823, 269.3533526642400778622, 272.6395283200241173749, 275.9220270102123290829, 279.200993965430128288, 282.4765665412116891156, 285.7488747641882258507, 289.0180418320454139153, 292.2841845718674106443, 295.547413860954322356, 298.8078350137381471417, 302.0655481380192927319, 305.3206484633933823289, 308.5732266444291723483, 311.8233690408870466893, 315.0711579770286534764, 318.3166719818574949839, 321.5599860119439830259, 324.8011716583234742954, 328.0402973388093942178, 331.2774284769334135822, 334.5126276686087502092, 337.7459548375092917198, 340.9774673800648711817, 344.2072203008903697386, 347.4352663393922341013, 350.6616560882294927891, 353.8864381042465695769, 357.1096590124413702462, 360.3313636034835973979, 363.5515949252544465667, 366.7703943688392440666, 369.9878017493687526755, 373.2038553820723982086, 376.418592153877207181, 379.632047590859482434, 382.8442559218319043728, 386.0552501383265858547, 389.2650620512144114178, 392.4737223441825629563, 395.6812606242753016321, 398.8877054696876857842, 402.0930844749878192283, 405.2974242939303204538, 408.5007506800118704543, 411.7030885249088362549, 414.9044618949269896699, 418.1048940655841664749, 421.3044075544382679762, 424.5030241522652300354, 427.7007649526844153739, 430.8976503803222705758},
{88.75200625787072627001, 95.23953245042417847799, 100.74870143787336473906, 105.76521227560629602746, 110.47141671358885162135, 114.95979972424866761671, 119.28500327024081484291, 123.48250458696664294163, 127.57687405914206625911, 131.58594947974446147258, 135.52315531918269083638, 139.39888573765937435827, 143.22137595350694132787, 146.99727602574898747947, 150.73204262210846860102, 154.43021474023356082209, 158.09561281430393794334, 161.731485713077151969, 165.3406213750709983038, 168.9254314939152663963, 172.4880173162617212728, 176.0302214500118848469, 179.5536691473898695602, 183.0598015574009035227, 186.5499027727151073863, 190.0251220256466896962, 193.486492052073989522, 196.9349443988401451795, 200.3713222715182908809, 203.796391386614124325, 207.210849192420165618, 210.6153327468608808051, 214.0104254824511598106, 217.3966630434158599102, 220.7745383448221760347, 224.1445059758762348177, 227.5069860475721168813, 230.8623675673456929605, 234.2110114092928302802, 237.5532529371156334455, 240.8894043276915161606, 244.2197566355794368518, 247.5445816325456103898, 250.8641334510421682967, 254.1786500562982918137, 257.4883545681197462133, 260.7934564505087761637, 264.0941525847075551856, 267.3906282391509661808, 270.6830579480207603271, 273.9716063085682373418, 277.2564287060718294867, 280.5376719741828056934, 283.815474997456720149, 287.089969262045454913, 290.3612793598142549619, 293.6295234505329994473, 296.8948136862568981939, 300.1572566015469705436, 303.416953472775085726, 306.6740006494036264493, 309.9284898598188917097, 313.1805084940241767565, 316.430139865257957821, 319.677463452390422898, 322.9225551247640182004, 326.1654873509775471267, 329.4063293929659390683, 332.6451474865967542672, 335.882005009887786245, 339.116962639846017585, 342.3500784988351645785, 345.5814082912957978146, 348.8110054315674067303, 352.0389211634947908347, 355.2652046724409395017, 358.4899031902743454644, 361.7130620938498180493, 364.9347249974577357216, 368.1549338396767891778, 371.3737289650291627987, 374.591149200804383874, 377.8072319293883822223, 381.0220131564073348567, 384.2355275749713439008, 387.4478086262806648837, 390.658888556836847987, 393.8687984724825819518, 397.0775683894770654843, 400.2852272827982190697, 403.4918031318488525276, 406.6973229637308963685, 409.9018128942398768961, 413.1052981667208663166, 416.3078031889170802809, 419.5093515679330457543, 422.7099661434257493329, 425.909669019129334763, 429.108481592811689458, 432.3064245847545899932},
{89.78406839841090594246, 96.29432425955627650183, 101.82131038027775105093, 106.85298700103625435531, 111.57258804424084530638, 116.07306157763444787179, 120.40933206516037046185, 124.6170657854369013067, 128.72096786904133380736, 132.73897666435348696723, 136.68459434764173448589, 140.56827679755446167374, 144.3983093489092738694, 148.18138350353311492372, 151.92299072520701215772, 155.6276996067881818568, 159.29935603492456402412, 162.94123097962594728259, 166.5561317329678005721, 170.1464870623910944911, 173.7144133768588696175, 177.261766828722487366, 180.7901848337278477221, 184.3011195167027527988, 187.7958649175568231726, 191.2755793194436790425, 194.741303723375618927, 198.1939772490095344647, 201.6344500617308055397, 205.0634942926500300229, 208.4818133177382632216, 211.8900496860452612473, 215.2887919284148273318, 218.6785804327919989647, 222.0599125368285627333, 225.4332469606406272973, 228.7990076804871316443, 232.1575873265045439289, 235.5093501734610386698, 238.8546347820331171793, 242.1937563387860061906, 245.5270087354152620294, 248.8546664215390280561, 252.1769860601516432421, 255.494208010550356103, 258.8065576599623203134, 262.1142466220973922039, 265.417473818328512778, 268.7164264550713205227, 272.0112809091300636081, 275.3022035312426918784, 278.5893513767492762398, 281.872872871187869175, 285.1529084176603931493, 288.4295909519832249634, 291.7030464509222218151, 294.973394398192883256, 298.2407482123688829698, 301.5052156403743887938, 304.7668991198273911579, 308.0258961131442359076, 311.2822994160025765174, 314.5361974424849829586, 317.7876744889833394005, 321.0368109787305605805, 324.2836836886373251733, 327.5283659599442687108, 330.7709278940516562057, 334.0114365347566063101, 337.2499560380104300743, 340.486547830203819836, 343.721270755893954486, 346.9541812158037544568, 350.1853332958483735457, 353.4147788878765532097, 356.6425678027538133998, 359.8687478763598470534, 363.0933650690232523286, 366.3164635588722887233, 369.5380858295401625175, 372.7582727526269773313, 375.977063665287524466, 379.1944964432841793628, 382.4106075698170000805, 385.6254322004184120291, 388.8390042241773624823, 392.0513563215373183086, 395.2625200188937647553, 398.4725257401997671038, 401.6814028557725244934, 404.8891797284795366099, 408.0958837574698943818, 411.3015414196041833742, 414.5061783087254528119, 417.7098191729035638866, 420.9124879497759070772, 424.1142078000988966086, 427.3150011396167450418, 430.514889669346732936, 433.7138944043734638765},
{90.81587723556425194462, 97.34869350840972301942, 102.89337071031470820335, 107.94011073906204473881, 112.67302193110717972844, 117.18551131258472253194, 121.53278329045656228655, 125.75069145422105520993, 129.86407443288758714868, 133.89097019124180951315, 137.84495788054230270231, 141.73655451398697674186, 145.57409506657849150994, 149.36431208775488362706, 153.11273150480108274111, 156.82395122259670533511, 160.50184233665840554058, 164.14969770686849164104, 167.770343794044689169, 171.3662262767739897141, 174.9394765823200200003, 178.4919642793230566536, 182.025338832463155925, 185.5410632395093030815, 189.0404413948897573315, 192.5246405487466093303, 195.9947098911591136904, 199.4515960453945709792, 202.8961560735259978672, 206.3291684635495492783, 209.7513424662197212603, 213.1633260731369135844, 216.5657128687807448798, 219.9590479436208558326, 223.3438330198581887472, 226.72053091334601944, 230.0895694330346544883, 233.4513448015532112848, 236.8062246662915473781, 240.1545507588212690355, 243.4966412511208644241, 246.8327928494030742124, 250.1632826600390225381, 253.4883698568652623677, 256.8082971748361941256, 260.1232922513789623571, 263.4335688337887655679, 266.7393278684639538651, 270.0407584856375584422, 273.3380388914465791731, 276.6313371776369771421, 279.9208120578856938124, 283.206613538594185774, 286.4888835310405619453, 289.7677564109443994576, 293.043359530778966379, 296.3158136895426705305, 299.5852335641607226156, 302.8517281062172333132, 306.1154009073071722756, 309.3763505359383097606, 312.6346708485982663701, 315.8904512773250375272, 319.143777095875675998, 322.3947296663728120699, 325.6433866681186127558, 328.8898223100974143039, 332.1341075285388474728, 335.3763101707804388674, 338.616495166550367208, 341.8547246876855116298, 345.0910582972056176265, 348.3255530885799950098, 351.5582638159474944064, 354.7892430159825761648, 358.0185411220392077696, 361.2462065711493328287, 364.4722859044030714628, 367.696823861193045896, 370.9198634677647550658, 374.1414461204782890015, 377.361611664153472444, 380.5803984658403996356, 383.7978434843299514533, 387.0139823356939911307, 390.2288493551222662538, 393.4424776553023807436, 396.654899181570343695, 399.8661447640419761693, 403.0762441669197051822, 406.2852261351548556836, 409.4931184386323405569, 412.6999479140325328735, 415.905740504513982716, 419.1105212973504220021, 422.3143145596461032116, 425.5171437722448681938, 428.7190316619403750059, 431.9200002320875641975, 435.120070791708667055},
{91.84743774077686399999, 98.40264832525614884036, 103.964892798768042461, 109.02659560016878906213, 113.77273188735402271333, 118.2971635973104946331, 122.6553725774968048731, 126.88339803393972892011, 131.00621087411634848808, 135.04194776094063375189, 139.00426410592540352777, 142.90373748579519134296, 146.74875204991090547601, 150.54608100878455395872, 154.30128442789621140969, 158.0189892470921708418, 161.70309153239349572175, 165.35690582668058840568, 168.9832775786011852806, 172.5846692186244079113, 176.1632270512828320879, 179.7208339359657154708, 183.2591512740147933796, 186.7796528353110994629, 190.2836522779056443762, 193.7723257367055195843, 197.2467305162449810674, 200.7078206755022800069, 204.1564601112843150482, 207.5934336118068230529, 211.019456250661924775, 214.4351814142828033907, 217.8412076968661711742, 221.2380848509114605937, 224.6263189457681964137, 228.0063768584297404643, 231.3786901984868529032, 234.7436587513280522634, 238.1016535093457596325, 241.4530193493198648714, 244.7980774047247091374, 248.1371271739959300243, 251.4704483994547558641, 254.7983027463496319286, 258.1209353071268529902, 261.4385759524159472047, 264.7514405471790430928, 268.0597320479202122707, 271.3636414946955788576, 274.663348909839033948, 277.9590241137659133019, 281.2508274668925568892, 284.5389105455761046285, 287.8234167590056638846, 291.1044819131379324706, 294.3822347270466313089, 297.6567973064283733571, 300.9282855784634275044, 304.1968096917561442889, 307.4624743846664513441, 310.7253793249822594059, 313.9856194235656306346, 317.2432851243270392065, 320.498462672636808675, 323.7512343640664198951, 327.0016787751610711062, 330.2498709777754047797, 333.4958827383539188076, 336.7397827034038647531, 339.9816365722893440189, 343.2215072583690641534, 346.4594550394052704817, 349.6955376980863846616, 352.9298106534296977039, 356.1623270837620660739, 359.3931380419150599367, 362.6222925632156375138, 365.8498377668034897232, 369.0758189507611184746, 372.300279681501955741, 375.5232618778249367902, 378.7448058900105020336, 381.9649505743026574763, 385.1837333630941547723, 388.4011903311067756019, 391.6173562578358704338, 394.8322646865074854794, 398.0459479797774150357, 401.2584373723841622352, 404.4697630209519212751, 407.6799540511251670504, 410.8890386022031271851, 414.09704386943020329, 417.3039961440872010868, 420.5099208515179313778, 423.7148425872162734678, 426.9187851510900754988, 430.1217715800102354976, 433.3238241787459025121, 436.5249645493799046213},
{92.87875473078962206818, 99.45619658996704970377, 105.03588670449291715589, 110.11245333592746815478, 114.87173103124047031646, 119.40803267709983970235, 123.77711511279747581275, 128.01520150323215794319, 132.14739384091723155205, 136.19192658863518087698, 140.16253071900673260187, 144.069843813702617444, 147.92229874074052068352, 151.72670899353954678726, 155.48866845747557480788, 159.21283283623939169194, 162.90312293306101321597, 166.56287477130857038529, 170.1949526103301817768, 173.8018354765110727927, 177.385684413511121676, 180.948395448472642629, 184.491641809374988129, 188.0169079393763534713, 191.5255171710290594793, 195.0186544433651086725, 198.4973851021783213458, 201.9626705755286171791, 205.4153815341443593051, 208.8563090108224000733, 212.286173850973286345, 215.7056347889820530045, 219.1152953856007651106, 222.5157100155493726525, 225.9073890585469537923, 229.290803418693976554, 232.6663884746831754458, 236.0345475453950991507, 239.3956549410294631, 242.750058658273471367, 246.0980827685313537248, 249.4400295404876005529, 252.7761813319025810785, 256.1068022802724693227, 259.4321398176129135407, 262.7524260309795762367, 266.0678788872849810321, 269.3787033384033259114, 272.6850923203873696979, 275.9872276587850141733, 279.2852808904817115453, 282.5794140111626491571, 285.8697801563484332691, 289.1565242229790152714, 292.4397834376775602585, 295.7196878770978900021, 298.9963609451286239326, 302.2699198111796700762, 305.5404758133001318326, 308.808134829460795908, 312.0729976199705623757, 315.3351601436772149272, 318.5947138503226623312, 321.8517459511759902689, 325.1063396698498989961, 328.3585744750135686044, 331.608526296544437916, 334.8562677265100137793, 338.1018682062362429182, 341.3453942005991072073, 344.5869093605691549594, 347.82647467494310472, 351.0641486121111079917, 354.2999872526315623241, 357.5340444133165057912, 360.7663717634687068736, 363.9970189338558105313, 367.2260336189566284581, 370.4534616729692691788, 373.6793472000297640765, 376.9037326390526937163, 380.1266588435716446991, 383.3481651569267681193, 386.5682894831189455315, 389.7870683536248125577, 393.0045369904438912308, 396.2207293656281151957, 399.4356782575248969574, 402.6494153039474050976, 405.8619710524697325647, 409.0733750080300024071, 412.2836556780110472573, 415.492840614955999286, 418.7009564570648357641, 421.908028966607549737, 425.1140830663800728466, 428.3191428743202934637, 431.5232317363934208296, 434.7263722578484841901, 437.928586332940870166},
{93.90983287419204403232, 100.50934594436101576775, 106.10636218721647135126, 111.19769535351916545096, 115.97003210188748312248, 120.51813239098957129506, 124.89802565534205155118, 129.14611739652336277665, 133.28763952429451821287, 137.3409234223946782308, 141.31977494047641434101, 145.2348911186066258585, 149.0947530975508691765, 152.90621428356606351222, 156.67490207040795834086, 160.40550066023403348031, 164.10195536509705317657, 167.76762349057100720265, 171.4053879332415694532, 175.0177441626424799735, 178.60686782622354648, 182.1746679983540635686, 185.7228296258109940044, 189.2528477281330558161, 192.7660552249780509828, 196.2636457804035205065, 199.7466927096466016999, 203.2161647444570773494, 206.6729392697850239382, 210.1178135083722970115, 213.5515140273332113628, 216.9747048629412730738, 220.3879945000821358367, 223.7919418965576160465, 227.1870617062857904517, 230.5738288269939142417, 233.9526823754406530101, 237.324029175188370065, 240.6882468274650672465, 244.0456864239435617348, 247.3966749507377609494, 250.7415174251223103429, 254.0804988000734791431, 257.4138856664336019059, 260.7419277781048523501, 264.0648594220116296752, 267.3829006515001312851, 270.696258399261509167, 274.0051274836852209753, 277.3096915207022710048, 280.6101237516076173315, 283.9065877960111796183, 287.1992383379200504938, 290.4882217519698309239, 293.773676675975023379, 297.0557345352360525911, 300.3345200234062312956, 303.6101515441712452482, 306.8827416175142799358, 310.1523972539214942179, 313.4192202995165265925, 316.6833077547918025584, 319.9447520693224165637, 323.2036414146000391461, 326.4600599369051768194, 329.7140879919423692911, 332.9658023627912749999, 336.2152764625742641932, 339.4625805231056931289, 342.7077817706673901183, 345.9509445899472432432, 349.1921306770815807721, 352.4313991826559244626, 355.6688068454414951942, 358.9044081175755331401, 362.1382552818311644634, 365.3703985615664189901, 368.6008862238913910024, 371.8297646765468345979, 375.0570785589461650483, 378.282870827795430506, 381.5071828376719114667, 384.7300544169112337322, 387.9515239391249212584, 391.1716283906448818435, 394.3904034341681568622, 397.607883468854150104, 400.8241016871072789151, 404.0390901282603837914, 407.2528797293581300653, 410.465500373224894103, 413.6769809339881182437, 416.8873493202157286263, 420.0966325158148349348, 423.3048566188284782788, 426.5120468782575795872, 429.7182277290263909723, 432.9234228252015988774, 436.1276550715677093939, 439.3309466536544077822},
{94.94067669762641514808, 101.56210380200755218365, 107.17632871967572327776, 112.2823327295174054437, 117.06764747425162479078, 121.62747618760643192993, 126.01811855304708927206, 130.27616082092722529615, 134.42696367525958760657, 138.48895456053604845807, 142.47601353393872392208, 146.39889655901697780418, 150.26613261284945537214, 154.08461465229814544796, 157.86000327454960399398, 161.59701092041322872962, 165.2996071871344589473, 168.9711704683096003522, 172.61460212785456066204, 176.2324139287868116715, 179.8267959897304738164, 183.3996703141541338522, 186.9527334619143357274, 190.4874909339178598108, 194.0052851512105720018, 197.507318425275614072, 200.9946719703202200715, 204.4683217575979891945, 207.9291518276657797575, 211.3779655395522273268, 214.8154951328437936145, 218.2424099004377234699, 221.6593232096548562473, 225.0667985628962477077, 228.4653548527040741124, 231.8554709374934757766, 235.2375896415455928727, 238.6121212647429306877, 241.979446672971515945, 245.3399200283408066248, 248.6938712087942581193, 252.0416079588484194092, 255.38341780675572821, 258.719569778062105883, 262.0503159311100810447, 265.3758927363516742732, 268.6965223182477130112, 272.0124135759338037392, 275.3237631966412835066, 278.6307565740042237838, 281.9335686418042987994, 285.2323646323578973931, 288.5273007675964870593, 291.8185248899009067596, 295.1061770388973825601, 298.3903899796864412731, 301.6712896873379369362, 304.9489957919314220033, 308.2236219879388051917, 311.495276411325328555, 314.7640619873766831773, 318.0300767519372304236, 321.2934141484605896557, 324.5541633030240144266, 327.8124092792375101332, 331.0682333147837057648, 334.3217130411517915576, 337.5729226879755496243, 340.8219332732492064932, 344.0688127805734296987, 347.3136263244754639652, 350.5564363047505872076, 353.7973025506854005508, 357.0362824559457651305, 360.2734311048424289397, 363.5088013906246476838, 366.7424441263956055195, 369.9744081491924931372, 373.2047404177280946724, 376.4334861042491372991, 379.6606886809289974784, 382.8863900011782204205, 386.1106303762253272972, 389.3334486472922331395, 392.5548822536629890585, 395.7749672969212396066, 398.9937386016105222674, 402.2112297725521286671, 405.4274732490375155216, 408.6425003560960363073, 411.8563413530239179969, 415.0690254793468018802, 418.2805809983756878177, 421.4910352385046634607, 424.7004146323882707555, 427.9087447541266775537, 431.1160503545779068451, 434.3223553949081619157, 437.5276830784837111633, 440.7320558812008058638},
{95.97129059166464060127, 102.61447735752180054241, 108.24579549913401876893, 113.36637622297411098771, 118.16458917335179990005, 122.73607714022786296937, 127.1374077584265336327, 131.40534647233624331861, 135.5653816212054891821, 139.63603586817085229009, 143.63126282254053635887, 147.56187684769193574355, 151.43645432975127078971, 155.26192742153933141896, 159.0439896250837429607, 162.7873813654201329367, 166.49609630596570722589, 170.17353373812884345143, 173.82261332669688092085, 177.4458629815170430869, 181.0454871624144458304, 184.6234206861587952587, 188.1813716220302791689, 191.7208558591232459412, 195.2432252357869931419, 198.7496906347898112442, 202.2413411001438030688, 205.7191597795953309501, 209.184037311750879028, 212.6367831392217375353, 216.0781351256969909225, 219.5087677762131028945, 222.9292992995348264802, 226.3402977048215692003, 229.7422860882477982346, 233.1357472365080106262, 236.5211276513452570759, 239.8988410810405581051, 243.2692716301690170952, 246.6327765070940024319, 249.9896884590428245249, 253.3403179367312362776, 256.6849550240273963747, 260.0238711627936048241, 263.3573206986001352373, 266.685542269299174637, 270.0087600553426161443, 273.3271849081168740682, 276.6410153703639792897, 279.9504386008907264106, 283.2556312141796238275, 286.5567600441604443835, 289.853982840241326463, 293.1474489027024468509, 296.4372996636975544783, 299.7236692193678207126, 303.0066848179308287832, 306.2864673080503329814, 309.5631315513073173903, 312.8367868021695062179, 316.1075370584860920622, 319.3754813852096793758, 322.6407142137620398496, 325.9033256192089361943, 329.1634015771874688408, 332.4210242023332739994, 335.6762719697811438663, 338.9292199211584113415, 342.1799398563532993351, 345.4285005122182715028, 348.6749677292594156659, 351.919404607265465397, 355.1618716507428495262, 358.4024269049449621086, 361.641126083213628106, 364.8780226862875956979, 368.1131681141760211609, 371.3466117711436310586, 374.578401164307938268, 377.8085819963070158623, 381.037198252458422708, 384.2642922827955090372, 387.4899048793361399631, 390.7140753489105329766, 393.9368415818491220468, 397.1582401168078786564, 400.3783062019871100671, 403.5970738529802135203, 406.8145759074710102147, 410.0308440769819524969, 413.2459089958605466774, 416.4598002676776323952, 419.6725465091985910851, 422.8841753920770164108, 426.0947136824097746641, 429.304187278282628685, 432.5126212454266188069, 435.7200398510971201539, 438.9264665962798654576, 442.1319242463211805444},
{97.00167881637850086918, 103.66647359538199966411, 109.31477145831433972489, 114.44983628785166283163, 119.26086888779398975657, 123.84394796110847853958, 128.25590684350141596161, 132.53368865074508903186, 136.70290828151066199672, 140.78218279298182451931, 144.78553870483402275608, 148.72384826751649842091, 152.60573485781482526401, 156.43816947720917596039, 160.22687824013883240406, 163.97662930666216200183, 167.69144019181528855058, 171.37473089846158634, 175.02943922914666970681, 178.6581090968157983917, 182.2629591750875376223, 185.8459369804976629703, 189.408761990098377884, 192.9529603897726435783, 196.479893352677231438, 199.9907802581459521431, 203.4867179121045986126, 206.9686965769263956734, 210.4376134327418096967, 213.8942839539716245104, 217.3394515808797312857, 220.7737959869195298096, 224.197940181999696828, 227.6124566448248718702, 231.0178726407807837963, 234.4146748529522914776, 237.803313430957149412, 241.1842055439849366517, 244.5577385097248534563, 247.9242725589712631469, 251.2841432860184657955, 254.6376638270393907798, 257.9851268021325252433, 261.326806051341057293, 264.6629581904808272881, 267.9938240088877287783, 271.3196297280743845192, 274.6405881376613419885, 277.9568996227322241917, 281.2687530948846416436, 284.5763268376519721365, 287.8797892756087104074, 291.1792996753058245566, 294.4750087851810830534, 297.7670594207267794614, 301.0555870004522733568, 304.3407200375334945439, 307.6225805914811868944, 310.9012846836717828068, 314.1769426801589766166, 317.4496596448115275808, 320.7195356654961556019, 323.986666155737313942, 327.2511421340327959244, 330.5130504827810105059, 333.7724741885784611553, 337.0294925654711581639, 340.2841814625885314975, 343.5366134574504330123, 346.7868580361149044914, 350.0349817612247099681, 353.2810484289126005011, 356.5251192154375194727, 359.7672528143452683237, 363.0075055648764919291, 366.2459315722812973913, 369.4825828206425917543, 372.7175092787586134202, 375.9507589995885245465, 379.182378213722787023, 382.4124114173018866313, 385.6409014547723784609, 388.8678895968378301237, 392.0934156139337089334, 395.3175178455293033877, 398.5402332655361290982, 401.7615975440807145063, 404.9816451058799872641, 408.2004091854395052199, 411.4179218792783332741, 414.6342141953693130078, 417.8493160999696752651, 421.0632565620042896784, 424.2760635951522243659, 427.487764297776609261, 430.698384890827972821, 433.9079507538421777101, 437.1164864591457474907, 440.3240158043736911551, 443.5305618433978392446},
{98.03184550662262570525, 104.71809929829906593568, 110.38326527578492126514, 115.53272308484023552853, 120.35649798263553122979, 124.95110101511540564525, 129.37362901399865568579, 133.66120127485195754784, 137.83955818241526635445, 141.92741038027226144083, 145.9388566699161312223, 149.88482668666441678483, 153.77399038817135362961, 157.61335728439420610243, 161.40868581572393638852, 165.16477163310017698135, 168.88565589295765520213, 172.57477912699537446739, 176.23509711565076461922, 179.8691696340724446528, 183.4792294447569283619, 187.0672366526701181324, 190.6349220429331504645, 194.1838220085514734285, 197.7153069765393988955, 201.2306047494600332701, 204.7308198285028433458, 208.2169495299182241682, 211.6898975198409896086, 215.1504852536367284011, 218.5994617014381819248, 222.0375116621381173399, 225.4652629071649547371, 228.8832923481695531217, 232.2921313858865962311, 235.692270568411208284, 239.0841636641126255534, 242.4682312360224487882, 245.8448637897557127523, 249.2144245550683144211, 252.5772519514279391076, 255.9336617800184795045, 259.2839491780541889727, 262.6283903658717118946, 265.9672442127775558578, 269.3007536438831556672, 272.6291469070224605039, 275.9526387162085151574, 279.2714312858579078573, 282.585715268124275591, 285.8956706040767583192, 289.2014672980895058622, 292.5032661236357171247, 295.8012192676727246564, 299.0954709199373405163, 302.3861578127215340582, 305.6734097160496362483, 308.9573498926147450106, 312.2380955163413622334, 315.5157580580130482352, 318.7904436410292099494, 322.0622533700265910507, 325.33128363481229074, 328.5976263918008376696, 331.8613694249234156854, 335.1225965877788743881, 338.3813880286203164656, 341.63782039961496505, 344.8919670516762080215, 348.1438982160430618463, 351.3936811736719535769, 354.6413804134070906038, 357.8870577798073885149, 361.1307726114287530399, 364.3725818702894121933, 367.6125402631820520813, 370.8507003554389217742, 374.0871126777041355828, 377.3218258262204971862, 380.5548865570957558352, 383.7863398749748014386, 387.0162291165094900539, 390.2445960289861905541, 393.4714808444424262346, 396.6969223495778585667, 399.9209579517410637446, 403.1436237412518544991, 406.3649545502990936173, 409.5849840086358477791, 412.8037445962771764733, 416.0212676933906940526, 419.2375836275561519639, 422.4527217185575450305, 425.6667103208595446166, 428.8795768639093075592, 432.0913478903948174477, 435.3020490925818071928, 438.5117053468429194666, 441.7203407464850218746, 444.9279786329734500689},
{99.06179467704789942673, 105.76936105516574571084, 111.45128538583030319842, 116.61504649259715659581, 121.45148751162797204723, 126.05754833271276993401, 130.49058712287995831502, 134.78789789597873874612, 138.97534547121148632276, 143.07173328732868188659, 147.09123181188463029943, 151.04482757308297296299, 154.94123670798527304359, 158.787506901740404938, 162.58942864001729689274, 166.35182482540358392804, 170.07876004971460568584, 173.77369519449232587538, 177.43960386135102573774, 181.0790615495029840006, 184.6943149878281640924, 188.2873367605240223839, 191.8598688629722662327, 195.4134578073204748397, 198.9494831949953489414, 202.4691811798002498712, 205.973663892747599203, 209.4639356443034032327, 212.9409065320684321336, 216.4054039423749967754, 219.858182329321271194, 223.2999315749894562069, 226.7312841733642432373, 230.1528214330444530025, 233.5650788567983363715, 236.9685508268496559378, 240.3636947016506848941, 243.7509344114238079805, 247.1306636249011969866, 250.5032485476779864709, 253.8690304028193854123, 257.2283276363650216299, 260.5814378837969692444, 263.9286397281022932928, 267.2701942755473784441, 270.6063465715166023878, 273.9373268756144381604, 277.263351812577852967, 280.5846254133065877909, 283.901340058421258453, 287.2136773351453961564, 290.5218088169304411372, 293.825896774063777439, 297.1260948224874862056, 300.4225485171834807391, 303.7153958957274421907, 307.0047679769615270188, 310.290789219169176754, 313.5735779416419835409, 316.8532467130979202441, 320.129902710033462224, 323.4036480477617166868, 326.6745800865982868595, 329.9427917154008387277, 333.2083716144426127907, 336.4714044994005072495, 339.7319713480614909938, 342.9901496111940990456, 346.2460134088921365522, 349.4996337135733281099, 352.7510785207046455626, 356.0004130082268253127, 359.2476996855617529235, 362.4929985330067368075, 365.7363671322481579519, 368.9778607886626465526, 372.2175326460159919996, 375.4554337941177323739, 378.6916133699421721624, 381.9261186526838957216, 385.1589951531771975181, 388.3902866980738132934, 391.6200355091415332765, 394.8482822780173767305, 398.0750662367227114675, 401.300425224223750541, 404.5243957492990181697, 407.747013049955440594, 410.9683111496165000165, 414.1883229102892258267, 417.4070800829015392447, 420.6246133549874831083, 423.8409523958850392299, 427.0561258995994551929, 430.2701616254741750491, 433.4830864368015081366, 436.6949263374959997282, 439.905706506945016618, 443.1154513331432670462, 446.3241844442107806183},
{100.09153022686196266022, 106.82026526861068183967, 112.5188399878384598557, 117.69681611844221436058, 122.54584822887447614903, 127.16330162233263784152, 131.60679368323883409175, 135.91379171134832428396, 140.11028392978548997022, 144.21516579713455997147, 148.24267884364793743064, 152.20386600833702590688, 156.10748921428156971712, 159.96063399522199920226, 163.76912260704191693914, 167.53780496950418025065, 171.2707689078639256787, 174.97149547803334465805, 178.64297594914847770701, 182.28780140902150707088, 185.9082324327738732484, 189.5062539767138302899, 193.0836191505180504151, 196.6418844991361871782, 200.1824387204270728495, 203.7065262487574035495, 207.2152667807002561455, 210.7096715623365897766, 214.1906570691519350928, 217.6590565693325910339, 221.1156299558224811857, 224.5610721523542975893, 227.9960203371515036046, 231.4210601803503310865, 234.8367312539725782382, 238.2435317439772662328, 241.6419225706769968522, 245.0323310052410119326, 248.4151538550824376756, 251.7907602788543028719, 255.1594942819557572915, 258.5216769354131257485, 261.877608354390778471, 265.2275694671239021204, 268.5718236005288986567, 271.910617904963308974, 275.2441846374375661254, 278.5727423199150303599, 281.8964967870858927198, 285.2156421360930161349, 288.5303615890655125623, 291.8408282779314984555, 295.147205959796299856, 298.4496496701545724472, 301.7483063203281129144, 305.0433152447638313878, 308.3348087031703674914, 311.622912341902099657, 314.9077456185032118794, 318.1894221928914593798, 321.4680502882824000831, 324.7437330246226014873, 328.0165687270083118001, 331.2866512113088788435, 334.5540700489871913959, 337.81891081290865989, 341.081255305752367827, 344.3411817724801110735, 347.5987650981786030198, 350.854076992465006144, 354.1071861615342933654, 357.3581584688271329401, 360.6070570852076283739, 363.8539426294601117472, 367.0988732998422231209, 370.3419049973667828746, 373.5830914414266663979, 376.8224842783243100986, 380.0601331832199894544, 383.2960859559700642215, 386.5303886112874986374, 389.7630854636217080681, 392.9942192071227801766, 396.2238309910260337745, 399.4519604907664153771, 402.6786459751081286641, 405.9039243695529112575, 409.127831316270307801, 412.3504012307749521886, 415.5716673555590987817, 418.791661810873283996, 422.010415642833922725, 425.227958869023729453, 428.4443205217389945981, 431.6595286890268464522, 434.873610553645601555, 438.0865924300720735517, 441.298499799671202294, 444.5093573441355177632, 447.7191889772947100325},
{101.12105594435195143888, 107.87081816218105245753, 113.58593705523246342182, 118.77804130854070763235, 123.63959059993568365859, 128.26837228216737097953, 132.72226088060119628194, 137.03889557675484620207, 141.24438698754704648897, 145.35772183147049801714, 149.39321211012358698374, 153.36195670084652864093, 157.27276292717356960223, 161.13275385131918152223, 164.94778322975991066828, 168.72272776957959174427, 172.46169833148919766725, 176.16819597371564076984, 179.84522948223329523912, 183.49540540059029422976, 187.120998032294092601, 190.7240046006633431896, 194.3061892354966559777, 197.869118429802056486, 201.4141899013165635883, 204.9426562955714623818, 208.4556448115866825998, 211.9541735734919616866, 215.4391653820102385613, 218.9114593389137507947, 222.3718207316379221642, 225.8209494847217675364, 229.259487422941619044, 232.6880245431355298056, 236.1071044543228928467, 239.5172291162828393812, 242.9188629834024554215, 246.3124366419533702512, 249.6983500139590524565, 253.076975188683902464, 256.4486589329043011213, 259.8137249230456668197, 263.1724757356273891346, 266.5251946269675630768, 269.872147128540495877, 273.2135824805771682274, 276.549734923313262835, 279.8808248626100025409, 283.2070599244106930399, 286.5286359105785591972, 289.8457376670308195815, 293.1585398736923992712, 296.4672077646013195079, 299.7718977854746490833, 303.0727581951625798525, 306.3699296166568560697, 309.6635455426602919763, 312.953732800151314546, 316.2406119778786986517, 319.5242978202862825352, 322.8048995909865028685, 326.0825214085675019113, 329.357262557224921291, 332.6292177744508588201, 335.8984775177841840636, 339.1651282124245193205, 342.4292524813332989176, 345.6909293592865053578, 348.9502344922024365149, 352.20724032294201921, 355.4620162646668798415, 358.7146288627399910188, 361.9651419460638267353, 365.2136167686703528758, 368.460112142304787234, 371.7046845606799512882, 374.9473883160193888376, 378.1882756084545289534, 381.4273966487933943912, 384.6647997551351490378, 387.9005314437656521589, 391.1346365147337133829, 394.3671581324755405111, 397.5981379018256059634, 400.8276159397255286147, 404.0556309429183110057, 407.2822202518931518053, 410.5074199113258601144, 413.7312647272414445936, 416.9537883211085693572, 420.1750231810601106689, 423.3950007104198799263, 426.613751273702579295, 429.8313042402421190273, 433.0476880255924532414, 436.2629301308349967584, 439.4770571799173911071, 442.6900949551398223803, 445.9020684308971934643, 449.1130018057781606564},
{102.15037551118398084782, 108.92102578717576477629, 114.6525843439735155016, 119.85873115760370743987, 124.73272481241529440159, 129.3727714114160882964, 133.83700058466299254532, 138.1632220186603165002, 142.37766773378010659697, 146.49941496343395667912, 150.54284560085799627132, 154.51911399854958859548, 158.43707250252248165304, 162.30388138963540270576, 166.12542565261545226165, 169.9066085604952945474, 173.65156381529891447943, 177.36381230883082112416, 181.04638019610702363744, 184.70188934607408623796, 188.3326276749928638524, 191.9406045700569157962, 195.5275950887578802596, 199.0951755889723304533, 202.6447527331505149492, 206.1775873098348674623, 209.6948139584978657743, 213.1974576247607086546, 216.6864473828475589283, 220.1626281206731410434, 223.6267704765577542357, 227.0795793356815419562, 230.5217011323053747249, 233.9537301556960481729, 237.3762140201276124197, 240.7896584297525812207, 244.1945313456748406038, 247.5912666438156778254, 250.9802673370970265437, 254.3619084232769089583, 257.7365394098537575391, 261.1044865593412043144, 264.4660548915404844239, 267.8215299739208368937, 271.1711795266369293034, 274.5152548648907440422, 277.853992198143953764, 281.1876138029940404505, 284.5163290842536819039, 287.8403355368459193553, 291.1598196194886737434, 294.4749575497435075551, 297.7859160288060308955, 301.092852903386895602, 304.3959177711464057168, 307.6952525353804490905, 310.9909919139924824476, 314.2832639072104670844, 317.5721902280062223054, 320.8578866987369485496, 324.1404636171456679789, 327.4200260945214303698, 330.6966743685248913639, 333.9705040929248121195, 337.2416066062614882678, 340.5100691812501059186, 343.7759752565571295951, 347.0394046524231205954, 350.3004337714633420817, 353.5591357858509534601, 356.8155808119746503584, 360.0698360735616379949, 363.3219660541664199808, 366.5720326398448105208, 369.8200952527597623774, 373.0662109764001084795, 376.3104346730343214049, 379.5528190939681830555, 382.7934149831271951203, 386.0322711744410938029, 389.2694346834684701674, 392.5049507936638081876, 395.738863137656854104, 398.9712137738847847369, 402.2020432588908489674, 405.4313907155787492889, 408.6592938976897722044, 411.8857892507493562831, 415.1109119697112167079, 418.334696053510157072, 421.5571743567191427622, 424.7783786384919509269, 427.9983396089596291838, 431.2170869732369805986, 434.4346494731842488069, 437.6510549270590169271, 440.8663302671839783149, 444.080501575747615166, 447.293594118846868333, 450.5056323788735405197},
{103.17949250649243413554, 109.97089402914964596374, 115.71878940065981369503, 120.93889451813316365794, 125.82526078605487714078, 130.47650982101558583687, 134.95102436049586087518, 139.28678324574898257454, 143.51013892944563792019, 147.64025842940957141509, 151.6915929620981498631, 155.67535190102116095344, 159.60043224405818201148, 163.47403117498301346778, 167.30206466355436567553, 171.08946231973249311, 174.84038049644137548701, 178.5583597535492266304, 182.24644347012190072462, 185.90726871262258446685, 189.5431368965943615927, 193.1560694718815875824, 196.7478523329373234965, 200.3200716208296799317, 203.8741428689100531535, 207.4113349417920612121, 210.9327898584978154222, 214.4395393305666554908, 217.9325186548769317894, 221.4125784588474759469, 224.8804946888071327392, 228.3369771510755623779, 231.7826768529337389689, 235.2181923423544758463, 238.6440752076257502469, 242.0608348682855463463, 245.4689427652164911422, 248.8688360389209611987, 252.2609207698594872952, 255.6455748424887750488, 259.0231504846703825593, 262.3939765259673358598, 265.7583604116395322957, 269.1165900036054214039, 272.4689351950338794582, 275.8156493613899633454, 279.1569706675411114678, 282.4931232478242965981, 285.8243182736896179756, 289.150754921599188105, 292.4726212522129925439, 295.7900950104876651159, 299.1033443551105386391, 302.4125285246576208442, 305.7177984469736762872, 309.0192972975033152157, 312.3171610116355709269, 315.6115187555456008094, 318.9024933595130790123, 322.1902017172568182162, 325.4747551544401162698, 328.7562597691636263527, 332.0348167469657198783, 335.3105226525888460151, 338.5834697005396013158, 341.8537460062661016459, 345.1214358195953655551, 348.3866197419128278015, 351.6493749284232688365, 354.9097752767051838673, 358.167891602657033029, 361.423791804832274097, 364.6775410180691587716, 367.9292017572397353867, 371.1788340518692661548, 374.4264955723113934817, 377.6722417481050528895, 380.9161258790856067624, 384.1581992397743286013, 387.3985111775266431749, 390.6371092048799316768, 393.8740390865058082588, 397.1093449211391810053, 400.3430692188267864153, 403.5752529738109301612, 406.8059357333396104203, 410.0351556626718051968, 413.2629496065262595837, 416.4893531472034236203, 419.7144006595930972913, 422.9381253632646851697, 426.1605593718226139282, 429.3817337396963001388, 432.6016785065219645852, 435.8204227392624748964, 439.0379945722011726617, 442.2544212449362250323, 445.4697291384933626799, 448.6839438096668613427, 451.8970900236912347429},
{104.20841041077154898176, 111.02042861410762952801, 116.78455957024467521461, 122.01854000923757641843, 126.91720818236523764789, 131.57959804388426562928, 136.06434347924977188031, 140.40959115996871715453, 144.64181301846601965022, 148.78026514051914677637, 152.83946750834392953076, 156.83068407107562316322, 160.76285611498892133526, 164.64321742896430742402, 168.47771470554672986987, 172.27130367882751906681, 176.02816316584029923764, 179.75185323213470550507, 183.44543433856072205984, 187.11155862360387488645, 190.752540890720504596, 194.3704145530413577874, 197.9669762529013869274, 201.5438218343563377842, 205.1023756291646008177, 208.6439145122536636778, 212.1695878223565018303, 215.6804339823171521023, 219.177394461688880502, 222.6613255815423382618, 226.1330085540520178949, 229.5931580678240740447, 233.042429667285707385, 236.4814261259315085138, 239.9107029753142949659, 243.3307733218190219325, 246.7421120595802570874, 250.1451595689896175307, 253.540324975031759365, 256.9279890273840726983, 260.3085066542033609915, 263.6822092333306733057, 267.0494066179072999143, 270.410388947825202948, 273.7654282738095464155, 277.1147800170722721714, 280.4586842842429729623, 283.7973670545640720733, 287.1310412540411087573, 290.4599077292927558144, 293.7841561321898567376, 297.103965724960028455, 300.4195061142247607616, 303.7309379213970222227, 307.0384133959723988921, 310.3420769774735898816, 313.6420658111382469712, 316.9385102218583138294, 320.2315341503723387233, 323.5212555552699088578, 326.807786783980292913, 330.0912349155778997113, 333.3717020779387567019, 336.6492857415193515931, 339.9240789917971517852, 343.1961707822068935851, 346.4656461692248686771, 349.7325865310919686459, 352.9970697715226317408, 356.2591705096188717131, 359.5189602570943562891, 362.7765075838113962521, 366.0318782725422756466, 369.2851354637843554869, 372.5363397913847332255, 375.7855495096639909674, 379.0328206126688881556, 382.2782069461300225268, 385.5217603126518583451, 388.763530570618541766, 392.003565727259094864, 395.2419120262794655813, 398.4786140304361244816, 401.7137146993960987774, 404.9472554632012164054, 408.1792762916296286526, 411.4098157597251494425, 414.6389111097443795589, 417.8665983097527845294, 421.0929121090836957064, 424.3178860908584533107, 427.5415527217514717986, 430.7639433991707599107, 433.9850884960122606855, 437.2050174031351921953, 440.423758569695279284, 443.6413395414632904278, 446.8577869972475601231, 450.0731267835311210415, 453.2873839474266329382},
{105.23713260958071202822, 112.06963511440689273968, 117.8499020033952575659, 123.09767602504220462833, 128.0085764138200090656, 132.68204634470569341811, 137.1769689283800062185, 141.53165736708691031908, 145.77270213851846958746, 149.91944769357862045344, 153.98648223340805779926, 157.98512384587976570521, 161.92435774912597497628, 165.81145404107340804846, 169.65238988763732016556, 173.45214693434691778753, 177.21492627907464425528, 180.94430733371262693155, 184.64336750127936008773, 188.3147738691101927771, 191.96085451925077786749, 195.5836547305636495785, 199.1849818057944845761, 202.7664412132174703373, 206.329466011787944003, 209.8753410221427385665, 213.405222843924657282, 216.9201565576054640299, 220.4210897562810665054, 223.908884409589290456, 227.384326954084402722, 230.8481369224400160881, 234.3009743609332331439, 237.7434462359230787035, 241.1761119919594414563, 244.5994883941759593703, 248.0140537638354007029, 251.4202516968961920822, 254.8184943401915394042, 258.2091652874526767393, 261.5926221473496821742, 264.9691988274931483175, 268.3392075715703664182, 271.7029407811937801964, 275.0606726493920037763, 278.4126606287966442143, 281.7591467543300240102, 285.1003588374665870069, 288.4365115468334605056, 291.7678073879600116667, 295.0944375933228034092, 298.4165829324126665312, 301.73441445033500773, 305.0480941424103847325, 308.3577755713429197697, 311.6636044327470309804, 314.9657190741497360277, 318.2642509720009906727, 321.559325170715251217, 324.8510606873228483237, 328.1395708849206956144, 331.4249638177706115848, 334.7073425505935679044, 337.9868054543439338532, 341.2634464805145303992, 344.5373554158169929483, 347.808618118899104371, 351.0773167405984246401, 354.343529929087149951, 357.6073330211344738268, 360.8687982205978857986, 364.127994765152175152, 367.3849890821729712115, 370.6398449346091976195, 373.8926235576047562207, 377.1433837865631345426, 380.392182177288616352, 383.6390731187836369876, 386.8841089392329226064, 390.1273400056608207493, 393.3688148177081713353, 396.6085800959387432803, 399.8466808650522842207, 403.0831605323512555655, 406.3180609617810474681, 409.5514225438386175537, 412.7832842616218325581, 416.0136837532710989463, 419.2426573710359558015, 422.4702402371819999498, 425.6964662969376666662, 428.9213683686658625735, 432.1449781914321178882, 435.3673264701286828937, 438.5884429183027395239, 441.8083562988265442123, 445.0270944625377825399, 448.2446843849696274234, 451.461152201281885354, 454.6765232394971305729},
{106.26566239707424735006, 113.11851895438334266981, 118.9148236635121374062, 124.17631074271636748563, 129.09937465263552996109, 133.783864729276730613, 138.28891142142399374367, 142.65299318678661347037, 146.90281813136327608877, 151.05781838158761652153, 155.13264982100898672528, 159.13868424760113975351, 163.08495046154770723068, 166.97875457934294247985, 170.82610399554725691109, 174.63200605842099115873, 178.40068396682478301707, 182.13573632261264366214, 185.8402573339328030185, 189.51692891605625001244, 193.16809232228384637256, 196.7958046014169097667, 200.4018836307067896251, 203.9879444252744880088, 207.5554287013136073432, 211.1056291616896664382, 214.6397096091663857756, 218.1587217290800512675, 221.6636191897637681041, 225.1552695650875995001, 228.6344644752007728573, 232.1019282592450854889, 235.5583254306160894684, 239.0042671163954885902, 242.4403166443337012904, 245.8669944106459686068, 249.2847821379945477881, 252.6941266139444983646, 256.0954429848345110041, 259.4891176675882931169, 262.8755109318880736473, 266.2549591968639006592, 269.6277770796515575318, 272.9942592275500666526, 276.3546819608406093688, 279.7093047494334490959, 283.0583715432459749495, 286.4021119734697072455, 289.7407424395657859678, 293.0744670948634441038, 296.4034787419645074659, 299.7279596477303842828, 303.0480822864064781577, 306.3640100183897452282, 309.6758977112412170893, 312.9838923087643625232, 316.288133353293575675, 319.588753465749351728, 322.8858787875048669707, 326.1796293876618223105, 329.4701196389423595983, 332.7574585650608640483, 336.041750162137952631, 339.3230936964533338525, 342.6015839805997510222, 345.8773116298928242094, 349.1503633007078041092, 352.4208219122510546672, 355.6887668531289177091, 358.9542741739472624664, 362.2174167670595667587, 365.4782645344781518686, 368.7368845448707559727, 371.9933411804817250687, 375.2476962747426303876, 378.5000092412701305582, 381.7503371948885480012, 384.998735065260188081, 388.2452557036572516572, 391.489949983364709595, 394.7328668941632215525, 397.9740536313046496653, 401.2135556793595503824, 404.4514168912858791626, 407.6876795630407065849, 410.9223845040317484912, 414.1555711036827150022, 417.3872773943656680077, 420.6175401109345517277, 423.846394747076654418, 427.0738756086828176653, 430.3000158644225954515, 433.5248475936971549754, 436.7484018321303944213, 439.9707086147474300116, 443.1917970169791862785, 446.411695193622228936, 449.630430415874136354, 452.8480291065565477983, 456.064516873630495026},
{107.29400297936561514689, 114.16708541571799098385, 119.97933133342853295744, 125.25445213013871637031, 130.18961183915916873622, 134.88506295344359386945, 139.40018140735162432975, 143.77360966232709002405, 148.03217255273102699694, 152.19538920377565184823, 156.27798265492054053366, 160.29137799361521733241, 164.24464725882607764056, 168.14513230055804623782, 171.99887050184887988227, 175.81089470885725936149, 179.58545004490591560316, 183.3261541483065007791, 187.03611789780441449846, 190.71803791788922788709, 194.3742685277194673919, 198.0068784519572602477, 201.6176960579798485776, 205.2083458317450551518, 208.7802780779457355322, 212.3347933192912741793, 215.8730625048646944739, 219.3961438729953273688, 222.9049971197552728694, 226.400495379644168554, 229.8834354162869146171, 233.35454633830012937, 236.8144970920188721105, 240.263902933610320663, 243.7033310446902513695, 247.1333054253108310036, 250.5543111741922565436, 253.966798246901273188, 257.3711847672652306795, 260.7678599548388142285, 264.1571867210911395927, 267.5395039786755727016, 270.9151287013128142133, 274.2843577661701842385, 277.6474696059293964644, 281.0047256938218263159, 284.3563718816316066266, 287.7026396079087484378, 291.0437469913052434478, 294.3798998219727769176, 297.7112924622812573167, 301.0381086666839569253, 304.360522329327643759, 307.6786981669517901231, 310.992792343712641043, 314.3029530437831522307, 317.6093209968998895504, 320.9120299614373446541, 324.2112071690757283677, 327.5069737346792079642, 330.7994450346085358066, 334.0887310563472842424, 337.3749367220178533149, 340.6581621880964438141, 343.9385031234005081106, 347.2160509672137210648, 350.4908931692287549722, 353.76311341282409237, 357.0327918230451833663, 360.3000051605302197192, 363.5648270025047278559, 366.8273279118654064548, 370.0875755952807002115, 373.3456350511522474735, 376.6015687082064649005, 379.8554365554181756858, 383.1072962639075077032, 386.357203301396546459, 389.6052110397627769197, 392.8513708561816180314, 396.0957322283098415544, 399.3383428239249291677, 402.5792485854020664412, 405.8184938093801519648, 409.0561212219406066254, 412.292172049597628191, 415.5266860863756066529, 418.7597017572284794577, 421.9912561780366695226, 425.2213852123997401082, 428.450123525426864832, 431.6775046347125100734, 434.9035609586712368119, 438.1283238623931382744, 441.3518237011700386656, 444.5740898618320967429, 447.7951508020248050338, 451.0150340875474780038, 454.2337664278661144912, 457.4513737099059416102},
{108.3221574777355292796, 115.21533964255780801977, 121.0434316218068310883, 126.33210795322032993226, 131.27929668988760509641, 135.98565053164781977955, 140.51078907951203064793, 144.8935175697914452258, 149.16077668179155781448, 153.33217187522761195982, 157.42249282870070230493, 161.44321750629343550497, 165.40346084883727130523, 169.31060016005893743886, 173.17070257573460353555, 176.98882623885408418568, 180.7692380239084327353, 184.51557445496005442826, 188.23096294925702470373, 191.91811472392849035553, 195.5793970604782018064, 199.2168902670211511571, 202.8324331181664695801, 206.427659496025670497, 210.0040282262408277142, 213.562847590049053793, 217.1052956270142786408, 220.6324370774577422006, 224.1452376184805519768, 227.644575902324582464, 231.1312537966215306377, 234.6060051430618912497, 238.0695032872817475683, 241.522367583390325873, 244.9651690379853277912, 248.3984352281249585286, 251.822654603625265658, 255.2382802647980749648, 258.6457332912626524235, 262.0454056849375402421, 265.4376629801244261112, 268.8228465652534269054, 272.2012757539966179088, 275.573249637783487148, 278.9390487470400117414, 282.298936544541868258, 285.6531607709785366789, 289.0019546600541640565, 292.345538038110979075, 295.6841183212774869469, 299.0178914214563531962, 302.3470425710266940978, 305.6717470749022410448, 308.9921709975275099346, 312.3084717914814375872, 315.6207988735693797464, 318.9292941536011424547, 322.2340925204591944736, 325.5353222895442873618, 328.833105615234395225, 332.1275588715979148293, 335.4187930042556120393, 338.7069138559812300948, 341.9920224683623529455, 345.2742153616062401133, 348.5535847943658151412, 351.8302190052752830453, 355.1042024377199554739, 358.3756159492181780023, 361.6445370066625412645, 364.9110398685508788871, 368.1751957552332314726, 371.437073008107529272, 374.6967372386129480689, 377.9542514677946171442, 381.2096762571456388081, 384.4630698313713702727, 387.7144881936658792045, 390.963985234040762912, 394.2116128312015448338, 397.4574209484261258062, 400.7014577238628257205, 403.9437695556320094715, 407.1844011820848005018, 410.4233957575446361814, 413.6607949238321367762, 416.8966388778506992669, 420.1309664354891709406, 423.3638150920787111654, 426.5952210796233395442, 429.825219421007539501, 433.0538439813694993203, 436.2811275168150031257, 439.5071017206345203986, 442.7317972671745838823, 445.9552438535040016458, 449.1774702390057382175, 452.3985042830163485748, 455.6183729806265910642, 458.8371024967492210821},
{109.3501289316924811601, 116.26328664640444578295, 122.1071309692487645821, 127.4092857829038637232, 132.36843770513428353297, 137.08563674510244569333, 141.62074438419710782599, 146.01272742694262898552, 150.28864153022597351071, 154.46817783610976443215, 158.56619215502061296728, 162.59421492239290534306, 166.56140365017687900515, 170.47517082115207999336, 174.34161309239932774569, 178.16581370633355546982, 181.9520611184638369109, 185.70401059061760327157, 189.42480594882344205259, 193.11717288835209018283, 196.78349155137548057493, 200.42585373868005913234, 204.0461085506604211208, 207.645899192191854798, 211.2266929434748707998, 214.7898057840005361324, 218.3364227889151563529, 221.8676151503803270034, 225.3843544805859096881, 228.8875249073275273295, 232.3779333634105030806, 235.8563183877775437289, 239.3233576922559847684, 242.7796746982369449068, 246.2258442088589510629, 249.6623973517607282293, 253.0898259032640055369, 256.508586085510676046, 259.9191019125292200932, 263.3217681486238854593, 266.7169529322407332352, 270.1050001100853177234, 273.4862313193737257974, 276.8609478504002013624, 280.2294323168714271578, 283.5919501575085906053, 286.9487509891096448412, 290.3000698284806251453, 293.6461281982940576741, 296.9871351299397745847, 300.3232880747382930483, 303.6547737334400000906, 306.9817688126943446458, 310.3044407161089012676, 313.6229481766011689382, 316.9374418359536376233, 320.2480647767961541273, 323.5549530116432312417, 326.8582359330945157068, 330.1580367288531148461, 333.454472764819573701, 336.7476559391711317617, 340.0376930100298083124, 343.3246858990532059855, 346.6087319730438585837, 349.8899243054623593776, 353.1683519195428590093, 356.444100014543785891, 359.7172501765192069192, 362.9878805748648609792, 366.25606614577561538, 369.5218787636462290915, 372.7853874013533915845, 376.0466582802727675281, 379.305755010809104207, 382.5627387241493794494, 385.8176681958876333899, 389.0705999621147925383, 392.3215884285168049695, 395.5706859729791852234, 398.8179430421551092318, 402.0634082424170552715, 405.3071284255782604739, 408.5491487697396029963, 411.7895128555896164666, 415.0282627384599193954, 418.2654390164151521044, 421.5010808946353383836, 424.7352262463292332669, 427.9679116703995075693, 431.1991725460643981484, 434.4290430846255805571, 437.6575563785583725816, 440.884744448087840687, 444.1106382854028555702, 447.3352678966495369275, 450.5586623418357592837, 453.7808497727693864349, 457.0018574691445950555, 460.2217118728829780454}
    };
}


// radial_basis_functions

// The schur decomposition from eigen is not working for some reason
// Eigen Schur decomposition is different form the numpy one and the diagonal elements are not the same
// So trying to implement the matrix square root using the eigenvalues
Eigen::MatrixXd matrixSqrt(const Eigen::MatrixXd& A) {
    // Ensure the matrix is square
    assert(A.rows() == A.cols());
    // using eigenvalues, A is also symmetric

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    Eigen::MatrixXd D = es.eigenvalues().asDiagonal();
    Eigen::MatrixXd V = es.eigenvectors();
    Eigen::MatrixXd sqrtD = D.cwiseSqrt();
    Eigen::MatrixXd sqrtA = V * sqrtD * V.transpose();
    return sqrtA;
}

// polynomial basis
void polynomial_basis(int n_max, double cutoff, int r_size, double *r, double *r_basis) {
    for (int i = 0; i < n_max; i++){
        int power = i + 3;
        for (int j = 0; j < r_size; j++){
            r_basis[i * r_size + j] = std::pow(cutoff - r[j],power);
        }
    }
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n_max, n_max);
    for (int i = 1; i < n_max + 1; i++){
        for (int j = 1; j < n_max + 1; j++){
            S(i - 1, j - 1) = (2 * std::pow((cutoff), (7 + i + j))) / (
                (5 + i + j) * (6 + i + j) * (7 + i + j)
            );
        }
    }
    S = S.inverse();
    S = matrixSqrt(S);


    Eigen::MatrixXd r_basis_map = Eigen::Map<Eigen::MatrixXd>(r_basis, r_size, n_max);
    Eigen::MatrixXd temp = r_basis_map * S;

    for (int i = 0; i < n_max; i++){
        for (int j = 0; j < r_size; j++){
            r_basis[i * r_size + j] = temp(j, i);
        }
    }
}


void bessel_basis(int n_max, double rc, int r_size, double *r, int r_basis_size, double *r_basis) {
    std::vector<double> r_vec(r, r + r_size);

    int u_all_size = (n_max + 2) * (n_max + 1);
    // This is a column major order, i.e. all the zeros for a given l are together
    // in a single column
    // n + 2 roots for n + 1 l values
    std::vector<double> u_all(u_all_size, 0.0);
    spherical_jn_zeros(n_max, u_all.data());
    int coeff_size = (n_max + 1) * (n_max + 1);
    std::vector<double> coeff_a(coeff_size, 0.0);
    std::vector<double> coeff_b(coeff_size, 0.0);

    for (int l = 0; l < n_max + 1; l++) {
        for (int n = 0; n < n_max - l + 1; n++) {
            double u_n  = u_all[l + (n_max + 1) * (n)];
            double u_n1 = u_all[l + (n_max + 1) * (n  + 1)];
            double c = std::sqrt(2 / (rc * rc * rc * (u_n * u_n + u_n1 * u_n1)));
            coeff_a[n * (n_max + 1) + l] = u_n1 / spherical_jn(l + 1, u_n) * c;
            coeff_b[n * (n_max + 1) + l] = -u_n / spherical_jn(l + 1, u_n1) * c;
        }
    }

    std::vector<double> e(n_max + 1, 0.0);
    std::vector<double> d(n_max + 1, 1.0);
    std::vector<double> gnl_mat(r_size * (n_max + 1) * (n_max + 1), 0.0);

    for (int l = 0; l < n_max + 1; l++) {
        for (int n = 1; n < n_max - l+1; n++) {
            double u_n   = u_all[l + (n_max + 1) * (n)];
            double u_n1  = u_all[l + (n_max + 1) * (n + 1)];
            double u_n_1 = u_all[l + (n_max + 1) * (n - 1)];
            e[n] = (u_n_1 * u_n_1 * u_n1 * u_n1) /
                    ((u_n_1 * u_n_1 + u_n * u_n) * (u_n * u_n + u_n1 * u_n1));
            d[n] = 1.0 - e[n] / d[n - 1];
        }

        for (int n = 0; n < n_max - l + 1; n++) {
            for (int a = 0; a < r_size; a++) {
                double u_n   = u_all[l + (n_max + 1) * (n)];
                double u_n1  = u_all[l + (n_max + 1) * (n + 1)];
                gnl_mat[a * (n_max + 1) * (n_max + 1) + n * (n_max + 1) + l] =
                        coeff_a[n * (n_max + 1) + l] * spherical_jn(l, u_n * r_vec[a] / rc) +
                        coeff_b[n * (n_max + 1) + l] * spherical_jn(l, u_n1 * r_vec[a] / rc);
            }
        }

        for (int n = 1; n < n_max - l + 1; n++) {
            for (int a = 0; a < r_size; a++) {
                gnl_mat[a * (n_max + 1) * (n_max + 1) + n * (n_max + 1) + l] =
                        (gnl_mat[a * (n_max + 1) * (n_max + 1) + n * (n_max + 1) + l] + std::sqrt(1.0 - d[n]) *
                         gnl_mat[a * (n_max + 1) * (n_max + 1) + (n - 1) * (n_max + 1) + l]) / std::sqrt(d[n]);
            }
        }
    }

    for (int i = 0; i < r_basis_size; i++) {
        r_basis[i] = gnl_mat[i];
    }
}

// spherical_harmonics
// factorials and double facotrials ************************************************

// The vast majority of SH evaluations will hit these precomputed values.
double factorial(int x) {
    static const double factorial_cache[FACT_CACHE_SIZE] = {1, 1, 2, 6, 24, 120, 720, 5040,
                                                            40320, 362880, 3628800, 39916800,
                                                            479001600, 6227020800,
                                                            87178291200, 1307674368000};

    if (x < FACT_CACHE_SIZE) {
        return factorial_cache[x];
    } else {
        double s = factorial_cache[FACT_CACHE_SIZE - 1];
        for (int n = FACT_CACHE_SIZE; n <= x; n++) {
            s *= n;
        }
        return s;
    }
}

double factorial(double x) {
    return factorial(static_cast<int>(x));
}

double double_factorial(int x) {
    static const double dbl_factorial_cache[FACT_CACHE_SIZE] = {1, 1, 2, 3, 8, 15, 48, 105,
                                                                384, 945, 3840, 10395, 46080,
                                                                135135, 645120, 2027025};

    if (x < FACT_CACHE_SIZE) {
        return dbl_factorial_cache[x];
    } else {
        double s = dbl_factorial_cache[FACT_CACHE_SIZE - (x % 2 == 0 ? 2 : 1)];
        double n = x;
        while (n >= FACT_CACHE_SIZE) {
            s *= n;
            n -= 2.0;
        }
        return s;
    }
}
// End factorials and double factorials ********************************************

// Misc math functions *************************************************************
// clamp the first argument to be greater than or equal to the second
// and less than or equal to the third.
double clamp(double val, double min, double max) {
    if (val < min) { val = min; }
    if (val > max) { val = max; }
    return val;
}

// Return true if the first value is within epsilon of the second value.
bool approx_equal(double actual, double expected) {
    double diff = std::abs(actual - expected);
    // 5 bits of error in mantissa (source of '32 *')
    return diff < 32 * std::numeric_limits<double>::epsilon();
}

// Return floating mod x % m.
double fmod(double x, double m) {
    return x - (m * floor(x / m));
}

std::array<double,3> sph2cart(double phi, double theta) {
    double r = sin(theta);
    return {r * cos(phi), r * sin(phi), cos(theta)};
}

std::array<double,3> cart2sph(const std::array<double,3> &r) {
    double norm = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    std::array<double, 3> r_norm = {r[0] / norm, r[1] / norm, r[2] / norm};
    // Explicitly clamp the z coordinate so that numeric errors don't cause it
    // to fall just outside of acos' domain.
    double const theta = acos(clamp(r_norm[2], -1.0, 1.0));
    // We don't need to divide r.y() or r.x() by sin(theta) since they are
    // both scaled by it and atan2 will handle it appropriately.
    double phi = atan2(r_norm[1], r_norm[0]);
    if (phi < 0.0) {
        phi += 2.0 * M_PI;
    }
    return {phi, theta, norm};
}


// *********************************************************************************

// Condon-Shortley phase: (-1)^m
double condon_shortley(int m) {
    return (m % 2 == 0) ? 1.0 : -1.0;
}

// Evaluate Associated Legendre Polynomial P(l,m) at x
// Evaluate the associated Legendre polynomial of degree @l and order @m at
// coordinate @x. The inputs must satisfy:
// 1. l >= 0
// 2. 0 <= m <= l
// 3. -1 <= x <= 1
// See http://en.wikipedia.org/wiki/Associated_Legendre_polynomials
//
double Plm(int l, int m, double theta) {
    // Compute Pmm(x) = (-1)^m(2m - 1)!!(1 - x^2)^(m/2), where !! is the double
    // factorial.
    double const x = cos(theta);
    double pmm = 1.0;
    // P00 is defined as 1.0, do don't evaluate Pmm unless we know m > 0
    if (m > 0) {
        double somx2 = sqrt((1.0 - x) * (1.0 + x));
        double fact = 1.0;
        for (int i = 1; i <= m; i++) {
            pmm *= (-fact) * somx2;
            fact += 2.0;
        }
    }

    if (l == m) {
        // Pml is the same as Pmm so there's no lifting to higher bands needed
        return pmm;
    }

    // Compute Pmm+1(x) = x(2m + 1)Pmm(x)
    double pmm1 = x * (2 * m + 1) * pmm;
    if (l == m + 1) {
        // Pml is the same as Pmm+1 so we are done as well
        return pmm1;
    }

    // Use the last two computed bands to lift up to the next band until l is
    // reached, using the recurrence relationship:
    // Pml(x) = (x(2l - 1)Pml-1 - (l + m - 1)Pml-2) / (l - m)
    for (int n = m + 2; n <= l; n++) {
        double pmn = (x * (2 * n - 1) * pmm1 - (n + m - 1) * pmm) / (n - m);
        pmm = pmm1;
        pmm1 = pmn;
    }
    // Pmm1 at the end of the above loop is equal to Pml
    return pmm1;
}

double kron(int i, int j) {
    if (i == j) {
        return 1.0;
    } else {
        return 0.0;
    }
}


// Normalization Factor for SH Function
double K(int l, int m) {
    double temp = ((2.0 * l + 1.0) * factorial(l - m)) / (4.0 * M_PI * factorial(l + m));
    return sqrt(temp);
}

// Evaluate SH Function at (theta,phi) **********************************************
std::array<double,2> Ylmi(int l, int m, double phi, double theta) {
    if (l < 0) {
        throw std::runtime_error("l must be >= 0");
    }
    if (!(-l <= m && m <= l)) {
        throw std::runtime_error("m must be between -l and l.");
    }

    if (l < 11){
        return Ylmi_analytical(l, m, theta, phi);
    }

    if (m < 0) {
        auto sh = Ylmi(l, -m, phi, theta);
        double phase = std::pow(-1.0, m);
        return {phase * sh[0], -phase * sh[1]};
    }
    double kml = sqrt((2.0 * l + 1) * factorial(l - m) /
                      (4.0 * M_PI * factorial(l + m)));
    double pml = Plm(l, m, theta);
    return {kml * cos(m * phi) * pml, kml * sin(m * phi) * pml};
}

double Ylm(int l, int m, double phi, double theta) {
//    double sign = (m % 2 == 0) ? 1.0 : -1.0;
    double sign = condon_shortley(m);
    auto sh = Ylmi(l, abs(m), phi, theta);
    if (m == 0) {
        return sh[0];
    }
    else if (m < 0) {
        return sign * M_SQRT2 * sh[1];
    }
    else {
        return sign * M_SQRT2 * sh[0];
    }
}

std::vector<double> Ylm_all_m(int l, double phi, double theta) {
    std::vector<double> sh(2 * l + 1);
    for (int m = -l; m <= l; m++) {
        sh[m + l] = (Ylm(l, m, phi, theta));
    }
    return sh;
}

std::vector<double> Ylmi_all_m(int l, double phi, double theta) {
    // first 2l+1 are real, second 2l+1 are imaginary
    int total_elem = (2 * l + 1);
    std::vector<double> sh(2 * total_elem);
    for (int m = -l; m <= l; m++) {
        auto sh_i = (Ylmi(l, m, phi, theta));
        sh[(m + l)] = sh_i[0];
        sh[(m + l) + total_elem] = sh_i[1];
    }
    return sh;
}

std::vector<double> Ylm_all_m_from_r(int l, const std::array<double,3> &r) {
    auto const sph = cart2sph(r);
    std::vector<double> sh(2 * l + 1);
    for (int m = -l; m <= l; m++) {
        sh[m + l] = Ylm(l, m, sph[0], sph[1]);
    }
    return sh;
}

std::vector<double> Ylmi_all_m_from_r(int l, const std::array<double,3> &r) {
    auto const sph = cart2sph(r);
    int total_elem = (2 * l + 1);
    std::vector<double> sh(2 * total_elem);
    for (int m = -l; m <= l; m++) {
        auto sh_i = (Ylmi(l, m, sph[0], sph[1]));
        sh[(m + l)] = sh_i[0];
        sh[(m + l) + total_elem] = sh_i[1];
    }
    return sh;
}

// pointer based version of above two functions
void Ylm_all_m(int l, double phi, double theta, double * sh) {
    for (int m = -l; m <= l; m++) {
        sh[m +l] = (Ylm(l, m, phi, theta));
    }
}

void Ylmi_all_m(int l, double phi, double theta, double * sh) {
    // first 2l+1 are real, second 2l+1 are imaginary
    int total_elem = (2 * l + 1);
    for (int m = -l; m <= l; m++) {
        auto sh_i = (Ylmi(l, m, phi, theta));
        sh[(m + l)] = sh_i[0];
        sh[(m + l) + total_elem] = sh_i[1];
    }
}

void Ylm_all_m_from_r(int l,  double * r, double * sh) {
    auto const sph = cart2sph({r[0], r[1], r[2]});
    for (int m = -l; m <= l; m++) {
        sh[m + l] = Ylm(l, m, sph[0], sph[1]);
    }
}

void Ylmi_all_m_from_r(int l, double * r, double * sh){
    auto const sph = cart2sph({r[0], r[1], r[2]});
    int total_elem = (2 * l + 1);
    for (int m = -l; m <= l; m++) {
        auto sh_i = (Ylmi(l, m, sph[0], sph[1]));
        sh[(m + l)] = sh_i[0];
        sh[(m + l) + total_elem] = sh_i[1];
    }
}

void Ylmi_all_l_from_r(int l_max, double * r, double * sh){
    auto const sph = cart2sph({r[0], r[1], r[2]});
    int total_elem = (l_max + 1) * (l_max + 1);
    for (int l = 0; l <= l_max; l++) {
        for (int m = -l; m <= l; m++) {
            auto sh_i = (Ylmi(l, m, sph[0], sph[1]));
            sh[(l * l + l + m)] = sh_i[0];
            sh[(l * l + l + m) + total_elem] = sh_i[1];
        }
    }
}

void Ylmi_all_l_from_r(int l_max, double * r, double * sh_real, double * sh_imag){
    // TODO: use pointer based fast sph harm version
    auto const sph = cart2sph({r[0], r[1], r[2]});
    int idx = 0;
    for (int l = 0; l <= l_max; l++) {
        for (int m = -l; m <= l; m++) {
            auto sh_i = (Ylmi(l, m, sph[0], sph[1]));
            sh_real[idx] = sh_i[0];
            sh_imag[idx] = sh_i[1];
            idx++;
        }
    }
}

void Ylm_all_l_from_r(int l_max, double * r, double * sh){
    auto const sph = cart2sph({r[0], r[1], r[2]});
    for (int l = 0; l <= l_max; l++) {
        for (int m = -l; m <= l; m++) {
            sh[(l * l + l + m)] = Ylm(l, m, sph[0], sph[1]);
        }
    }
}


// *********************************************************************************
// Analytical SH functions *********************************************************
// *********************************************************************************

std::array<double, 2> Y_0_0(double theta, double phi){
    double sph = 1.0/2 * std::sqrt(1/(M_PI) );
    return {sph * 1.0, 0.0};
}


std::array<double, 2> Y_1_neg1(double theta, double phi){
    double sph = 1.0/2 * std::sqrt(3/(2 * M_PI) ) * sin(theta);
    return {sph * cos(-phi), sph * sin(-phi)};
}
std::array<double, 2> Y_1_0(double theta, double phi){
    double sph = 1.0/2 * std::sqrt(3/(M_PI) ) * cos(theta);
    return {sph * 1.0, 0.0};
}
std::array<double, 2> Y_1_1(double theta, double phi){
    double sph = -1.0/2 * std::sqrt(3/(2 * M_PI) ) * sin(theta);
    return {sph * cos(phi), sph * sin(phi)};
}


std::array<double, 2> Y_2_neg2(double theta, double phi){
    double sph = 1.0/4 * std::sqrt(15/(2 * M_PI) ) * std::pow(sin(theta), 2);
    return {sph * cos(-2 * phi), sph * sin(-2 * phi)};
}
std::array<double, 2> Y_2_neg1(double theta, double phi){
    double sph = 1.0/2 * std::sqrt(15/(2 * M_PI) ) * sin(theta) * cos(theta);
    return {sph * cos(-phi), sph * sin(-phi)};
}
std::array<double, 2> Y_2_0(double theta, double phi){
    double sph = 1.0/4 * std::sqrt(5/(M_PI) ) * (3 * std::pow(cos(theta) ,2) - 1) ;
    return {sph * 1.0, 0.0};
}
std::array<double, 2> Y_2_1(double theta, double phi){
    double sph = -1.0/2 * std::sqrt(15/(2 * M_PI) ) * sin(theta) * cos(theta);
    return {sph * cos(phi), sph * sin(phi)};
}
std::array<double, 2> Y_2_2(double theta, double phi){
    double sph = 1.0/4 * std::sqrt(15/(2 * M_PI) ) * std::pow(sin(theta), 2);
    return {sph * cos(2 * phi), sph * sin(2 * phi)};
}


std::array<double, 2> Y_3_neg3(double theta, double phi){
    double sph = 1.0/8 * std::sqrt(35/(M_PI) ) * std::pow(sin(theta), 3) ;
    return {sph * cos(-3 * phi), sph * sin(-3 * phi)};
}
std::array<double, 2> Y_3_neg2(double theta, double phi){
    double sph = 1.0/4 * std::sqrt(105/(2 * M_PI) ) * std::pow(sin(theta), 2) * cos(theta) ;
    return {sph * cos(-2 * phi), sph * sin(-2 * phi)};
}
std::array<double, 2> Y_3_neg1(double theta, double phi){
    double sph = 1.0/8 * std::sqrt(21/(M_PI) ) * sin(theta) * (5* std::pow(cos(theta) ,2) - 1) ;
    return {sph * cos(-phi), sph * sin(-phi)};
}
std::array<double, 2> Y_3_0(double theta, double phi){
    double sph = 1.0/4 * std::sqrt(7/(M_PI) ) * (5* std::pow(cos(theta) ,3) - 3* cos(theta)) ;
    return {sph * 1.0, 0.0};
}
std::array<double, 2> Y_3_1(double theta, double phi){
    double sph = -1.0/8 * std::sqrt(21/(M_PI) ) * sin(theta) * (5* std::pow(cos(theta) ,2) - 1) ;
    return {sph * cos(phi), sph * sin(phi)};
}
std::array<double, 2> Y_3_2(double theta, double phi){
    double sph = 1.0/4 * std::sqrt(105/(2 * M_PI) ) * std::pow(sin(theta), 2) * cos(theta) ;
    return {sph * cos(2 * phi), sph * sin(2 * phi)};
}
std::array<double, 2> Y_3_3(double theta, double phi){
    double sph = -1.0/8 * std::sqrt(35/(M_PI) ) * std::pow(sin(theta), 3) ;
    return {sph * cos(3 * phi), sph * sin(3 * phi)};
}



std::array<double, 2> Y_4_neg4(double theta, double phi){
    double sph = 3.0/16 * std::sqrt(35/(2 * M_PI) ) * std::pow(sin(theta), 4) ;
    return {sph * cos(-4 * phi), sph * sin(-4 * phi)};
}
std::array<double, 2> Y_4_neg3(double theta, double phi){
    double sph = 3.0/8 * std::sqrt(35/(M_PI) ) * std::pow(sin(theta), 3) * cos(theta) ;
    return {sph * cos(-3 * phi), sph * sin(-3 * phi)};
}
std::array<double, 2> Y_4_neg2(double theta, double phi){
    double sph = 3.0/8 * std::sqrt(5/(2 * M_PI) ) * std::pow(sin(theta), 2) * (7* std::pow(cos(theta) ,2) - 1) ;
    return {sph * cos(-2 * phi), sph * sin(-2 * phi)};
}
std::array<double, 2> Y_4_neg1(double theta, double phi){
    double sph = 3.0/8 * std::sqrt(5/(M_PI) ) * sin(theta) * (7* std::pow(cos(theta) ,3) - 3* cos(theta)) ;
    return {sph * cos(-phi), sph * sin(-phi)};
}
std::array<double, 2> Y_4_0(double theta, double phi){
    double sph = 3.0/16 * std::sqrt(1/(M_PI) ) * (35* std::pow(cos(theta) ,4) - 30* std::pow(cos(theta) ,2) + 3) ;
    return {sph * 1.0, 0.0};
}
std::array<double, 2> Y_4_1(double theta, double phi){
    double sph = -3.0/8 * std::sqrt(5/(M_PI) ) * sin(theta) * (7* std::pow(cos(theta) ,3) - 3* cos(theta)) ;
    return {sph * cos(phi), sph * sin(phi)};
}
std::array<double, 2> Y_4_2(double theta, double phi){
    double sph = 3.0/8 * std::sqrt(5/(2 * M_PI) ) * std::pow(sin(theta), 2) * (7* std::pow(cos(theta) ,2) - 1) ;
    return {sph * cos(2 * phi), sph * sin(2 * phi)};
}
std::array<double, 2> Y_4_3(double theta, double phi){
    double sph = -3.0/8 * std::sqrt(35/(M_PI) ) * std::pow(sin(theta), 3) * cos(theta) ;
    return {sph * cos(3 * phi), sph * sin(3 * phi)};
}
std::array<double, 2> Y_4_4(double theta, double phi){
    double sph = 3.0/16 * std::sqrt(35/(2 * M_PI) ) * std::pow(sin(theta), 4) ;
    return {sph * cos(4 * phi), sph * sin(4 * phi)};
}



std::array<double, 2> Y_5_neg5(double theta, double phi){
    double sph = 3.0/32 * std::sqrt(77/(M_PI) ) * std::pow(sin(theta), 5);
    return {sph * cos(-5 * phi), sph * sin(-5 * phi)};
}
std::array<double, 2> Y_5_neg4(double theta, double phi){
    double sph = 3.0/16 * std::sqrt(385/(2 * M_PI) ) * std::pow(sin(theta), 4) * cos(theta);
    return {sph * cos(-4 * phi), sph * sin(-4 * phi)};
}
std::array<double, 2> Y_5_neg3(double theta, double phi){
    double sph = 1.0/32 * std::sqrt(385/(M_PI) ) * std::pow(sin(theta), 3) * (9* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(-3 * phi), sph * sin(-3 * phi)};
}
std::array<double, 2> Y_5_neg2(double theta, double phi){
    double sph = 1.0/8 * std::sqrt(1155/(2 * M_PI) ) * std::pow(sin(theta), 2) * (3* std::pow(cos(theta) ,3) - cos(theta));
    return {sph * cos(-2 * phi), sph * sin(-2 * phi)};
}
std::array<double, 2> Y_5_neg1(double theta, double phi){
    double sph = 1.0/16 * std::sqrt(165/(2 * M_PI) ) * sin(theta) * (21* std::pow(cos(theta) ,4) - 14* std::pow(cos(theta) ,2) + 1);
    return {sph * cos(-phi), sph * sin(-phi)};
}
std::array<double, 2> Y_5_0(double theta, double phi){
    double sph = 1.0/16 * std::sqrt(11/(M_PI) ) * (63* std::pow(cos(theta) ,5) - 70* std::pow(cos(theta) ,3) + 15* cos(theta));
    return {sph * 1.0, 0.0};
}
std::array<double, 2> Y_5_1(double theta, double phi){
    double sph = -1.0/16 * std::sqrt(165/(2 * M_PI) ) * sin(theta) * (21* std::pow(cos(theta) ,4) - 14* std::pow(cos(theta) ,2) + 1);
    return {sph * cos(phi), sph * sin(phi)};
}
std::array<double, 2> Y_5_2(double theta, double phi){
    double sph = 1.0/8 * std::sqrt(1155/(2 * M_PI) ) * std::pow(sin(theta), 2) * (3* std::pow(cos(theta) ,3) - cos(theta));
    return {sph * cos(2 * phi), sph * sin(2 * phi)};
}
std::array<double, 2> Y_5_3(double theta, double phi){
    double sph = -1.0/32 * std::sqrt(385/(M_PI) ) * std::pow(sin(theta), 3) * (9* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(3 * phi), sph * sin(3 * phi)};
}
std::array<double, 2> Y_5_4(double theta, double phi){
    double sph = 3.0/16 * std::sqrt(385/(2 * M_PI) ) * std::pow(sin(theta), 4) * cos(theta);
    return {sph * cos(4 * phi), sph * sin(4 * phi)};
}
std::array<double, 2> Y_5_5(double theta, double phi){
    double sph = -3.0/32 * std::sqrt(77/(M_PI) ) * std::pow(sin(theta), 5);
    return {sph * cos(5 * phi), sph * sin(5 * phi)};
}



std::array<double, 2> Y_6_neg6(double theta, double phi){
    double sph = 1.0/64 * std::sqrt(3003/(M_PI) ) * std::pow(sin(theta), 6);
    return {sph * cos(-6 * phi), sph * sin(-6 * phi)};
}
std::array<double, 2> Y_6_neg5(double theta, double phi){
    double sph = 3.0/32 * std::sqrt(1001/(M_PI) ) * std::pow(sin(theta), 5) * cos(theta);
    return {sph * cos(-5 * phi), sph * sin(-5 * phi)};
}
std::array<double, 2> Y_6_neg4(double theta, double phi){
    double sph = 3.0/32 * std::sqrt(91/(2 * M_PI) ) * std::pow(sin(theta), 4) * (11* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(-4 * phi), sph * sin(-4 * phi)};
}
std::array<double, 2> Y_6_neg3(double theta, double phi){
    double sph = 1.0/32 * std::sqrt(1365/(M_PI) ) * std::pow(sin(theta), 3) * (11* std::pow(cos(theta) ,3) - 3* cos(theta));
    return {sph * cos(-3 * phi), sph * sin(-3 * phi)};
}
std::array<double, 2> Y_6_neg2(double theta, double phi){
    double sph = 1.0/64 * std::sqrt(1365/(M_PI) ) * std::pow(sin(theta), 2) * (33* std::pow(cos(theta) ,4) - 18* std::pow(cos(theta) ,2) + 1);
    return {sph * cos(-2 * phi), sph * sin(-2 * phi)};
}
std::array<double, 2> Y_6_neg1(double theta, double phi){
    double sph = 1.0/16 * std::sqrt(273/(2 * M_PI) ) * sin(theta) * (33* std::pow(cos(theta) ,5) - 30* std::pow(cos(theta) ,3) + 5* cos(theta));
    return {sph * cos(-phi), sph * sin(-phi)};
}
std::array<double, 2> Y_6_0(double theta, double phi){
    double sph = 1.0/32 * std::sqrt(13/(M_PI) ) * (231* std::pow(cos(theta) ,6) - 315* std::pow(cos(theta) ,4) + 105* std::pow(cos(theta) ,2) - 5);
    return {sph * 1.0, 0.0};
}
std::array<double, 2> Y_6_1(double theta, double phi){
    double sph = -1.0/16 * std::sqrt(273/(2 * M_PI) ) * sin(theta) * (33* std::pow(cos(theta) ,5) - 30* std::pow(cos(theta) ,3) + 5* cos(theta));
    return {sph * cos(phi), sph * sin(phi)};
}
std::array<double, 2> Y_6_2(double theta, double phi){
    double sph = 1.0/64 * std::sqrt(1365/(M_PI) ) * std::pow(sin(theta), 2) * (33* std::pow(cos(theta) ,4) - 18* std::pow(cos(theta) ,2) + 1);
    return {sph * cos(2 * phi), sph * sin(2 * phi)};
}
std::array<double, 2> Y_6_3(double theta, double phi){
    double sph = -1.0/32 * std::sqrt(1365/(M_PI) ) * std::pow(sin(theta), 3) * (11* std::pow(cos(theta) ,3) - 3* cos(theta));
    return {sph * cos(3 * phi), sph * sin(3 * phi)};
}
std::array<double, 2> Y_6_4(double theta, double phi){
    double sph = 3.0/32 * std::sqrt(91/(2 * M_PI) ) * std::pow(sin(theta), 4) * (11* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(4 * phi), sph * sin(4 * phi)};
}
std::array<double, 2> Y_6_5(double theta, double phi){
    double sph = -3.0/32 * std::sqrt(1001/(M_PI) ) * std::pow(sin(theta), 5) * cos(theta);
    return {sph * cos(5 * phi), sph * sin(5 * phi)};
}
std::array<double, 2> Y_6_6(double theta, double phi){
    double sph = 1.0/64 * std::sqrt(3003/(M_PI) ) * std::pow(sin(theta), 6);
    return {sph * cos(6 * phi), sph * sin(6 * phi)};
}



std::array<double, 2> Y_7_neg7(double theta, double phi){
    double sph = 3.0/64 * std::sqrt(715/(2 * M_PI) ) * std::pow(sin(theta), 7);
    return {sph * cos(-7 * phi), sph * sin(-7 * phi)};
}
std::array<double, 2> Y_7_neg6(double theta, double phi){
    double sph = 3.0/64 * std::sqrt(5005/(M_PI) ) * std::pow(sin(theta), 6) * cos(theta);
    return {sph * cos(-6 * phi), sph * sin(-6 * phi)};
}
std::array<double, 2> Y_7_neg5(double theta, double phi){
    double sph = 3.0/64 * std::sqrt(385/(2 * M_PI) ) * std::pow(sin(theta), 5) * (13* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(-5 * phi), sph * sin(-5 * phi)};
}
std::array<double, 2> Y_7_neg4(double theta, double phi){
    double sph = 3.0/32 * std::sqrt(385/(2 * M_PI) ) * std::pow(sin(theta), 4) * (13* std::pow(cos(theta) ,3) - 3* cos(theta));
    return {sph * cos(-4 * phi), sph * sin(-4 * phi)};
}
std::array<double, 2> Y_7_neg3(double theta, double phi){
    double sph = 3.0/64 * std::sqrt(35/(2 * M_PI) ) * std::pow(sin(theta), 3) * (143* std::pow(cos(theta) ,4) - 66* std::pow(cos(theta) ,2) + 3);
    return {sph * cos(-3 * phi), sph * sin(-3 * phi)};
}
std::array<double, 2> Y_7_neg2(double theta, double phi){
    double sph = 3.0/64 * std::sqrt(35/(M_PI) ) * std::pow(sin(theta), 2) * (143* std::pow(cos(theta) ,5) - 110* std::pow(cos(theta) ,3) + 15* cos(theta));
    return {sph * cos(-2 * phi), sph * sin(-2 * phi)};
}
std::array<double, 2> Y_7_neg1(double theta, double phi){
    double sph = 1.0/64 * std::sqrt(105/(2 * M_PI) ) * sin(theta) * (429* std::pow(cos(theta) ,6) - 495* std::pow(cos(theta) ,4) + 135* std::pow(cos(theta) ,2) - 5);
    return {sph * cos(-phi), sph * sin(-phi)};
}
std::array<double, 2> Y_7_0(double theta, double phi){
    double sph = 1.0/32 * std::sqrt(15/(M_PI) ) * (429* std::pow(cos(theta) ,7) - 693* std::pow(cos(theta) ,5) + 315* std::pow(cos(theta) ,3) - 35* cos(theta));
    return {sph * 1.0, 0.0};
}
std::array<double, 2> Y_7_1(double theta, double phi){
    double sph = -1.0/64 * std::sqrt(105/(2 * M_PI) ) * sin(theta) * (429* std::pow(cos(theta) ,6) - 495* std::pow(cos(theta) ,4) + 135* std::pow(cos(theta) ,2) - 5);
    return {sph * cos(phi), sph * sin(phi)};
}
std::array<double, 2> Y_7_2(double theta, double phi){
    double sph = 3.0/64 * std::sqrt(35/(M_PI) ) * std::pow(sin(theta), 2) * (143* std::pow(cos(theta) ,5) - 110* std::pow(cos(theta) ,3) + 15* cos(theta));
    return {sph * cos(2 * phi), sph * sin(2 * phi)};
}
std::array<double, 2> Y_7_3(double theta, double phi){
    double sph = -3.0/64 * std::sqrt(35/(2 * M_PI) ) * std::pow(sin(theta), 3) * (143* std::pow(cos(theta) ,4) - 66* std::pow(cos(theta) ,2) + 3);
    return {sph * cos(3 * phi), sph * sin(3 * phi)};
}
std::array<double, 2> Y_7_4(double theta, double phi){
    double sph = 3.0/32 * std::sqrt(385/(2 * M_PI) ) * std::pow(sin(theta), 4) * (13* std::pow(cos(theta) ,3) - 3* cos(theta));
    return {sph * cos(4 * phi), sph * sin(4 * phi)};
}
std::array<double, 2> Y_7_5(double theta, double phi){
    double sph = -3.0/64 * std::sqrt(385/(2 * M_PI) ) * std::pow(sin(theta), 5) * (13* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(5 * phi), sph * sin(5 * phi)};
}
std::array<double, 2> Y_7_6(double theta, double phi){
    double sph = 3.0/64 * std::sqrt(5005/(M_PI) ) * std::pow(sin(theta), 6) * cos(theta);
    return {sph * cos(6 * phi), sph * sin(6 * phi)};
}
std::array<double, 2> Y_7_7(double theta, double phi){
    double sph = -3.0/64 * std::sqrt(715/(2 * M_PI) ) * std::pow(sin(theta), 7);
    return {sph * cos(7 * phi), sph * sin(7 * phi)};
}



std::array<double, 2> Y_8_neg8(double theta, double phi){
    double sph = 3.0/256 * std::sqrt(12155/(2 * M_PI) ) * std::pow(sin(theta), 8);
    return {sph * cos(-8 * phi), sph * sin(-8 * phi)};
}
std::array<double, 2> Y_8_neg7(double theta, double phi){
    double sph = 3.0/64 * std::sqrt(12155/(2 * M_PI) ) * std::pow(sin(theta), 7) * cos(theta);
    return {sph * cos(-7 * phi), sph * sin(-7 * phi)};
}
std::array<double, 2> Y_8_neg6(double theta, double phi){
    double sph = 1.0/128 * std::sqrt(7293/(M_PI) ) * std::pow(sin(theta), 6) * (15* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(-6 * phi), sph * sin(-6 * phi)};
}
std::array<double, 2> Y_8_neg5(double theta, double phi){
    double sph = 3.0/64 * std::sqrt(17017/(2 * M_PI) ) * std::pow(sin(theta), 5) * (5* std::pow(cos(theta) ,3) - cos(theta));
    return {sph * cos(-5 * phi), sph * sin(-5 * phi)};
}
std::array<double, 2> Y_8_neg4(double theta, double phi){
    double sph = 3.0/128 * std::sqrt(1309/(2 * M_PI) ) * std::pow(sin(theta), 4) * (65* std::pow(cos(theta) ,4) - 26* std::pow(cos(theta) ,2) + 1);
    return {sph * cos(-4 * phi), sph * sin(-4 * phi)};
}
std::array<double, 2> Y_8_neg3(double theta, double phi){
    double sph = 1.0/64 * std::sqrt(19635/(2 * M_PI) ) * std::pow(sin(theta), 3) * (39* std::pow(cos(theta) ,5) - 26* std::pow(cos(theta) ,3) + 3* cos(theta));
    return {sph * cos(-3 * phi), sph * sin(-3 * phi)};
}
std::array<double, 2> Y_8_neg2(double theta, double phi){
    double sph = 3.0/128 * std::sqrt(595/(M_PI) ) * std::pow(sin(theta), 2) * (143* std::pow(cos(theta) ,6) - 143* std::pow(cos(theta) ,4) + 33* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(-2 * phi), sph * sin(-2 * phi)};
}
std::array<double, 2> Y_8_neg1(double theta, double phi){
    double sph = 3.0/64 * std::sqrt(17/(2 * M_PI) ) * sin(theta) * (715* std::pow(cos(theta) ,7) - 1001* std::pow(cos(theta) ,5) + 385* std::pow(cos(theta) ,3) - 35* cos(theta));
    return {sph * cos(-phi), sph * sin(-phi)};
}
std::array<double, 2> Y_8_0(double theta, double phi){
    double sph = 1.0/256 * std::sqrt(17/(M_PI) ) * (6435* std::pow(cos(theta) ,8) - 12012* std::pow(cos(theta) ,6) + 6930* std::pow(cos(theta) ,4) - 1260* std::pow(cos(theta) ,2) + 35);
    return {sph * 1.0, 0.0};
}
std::array<double, 2> Y_8_1(double theta, double phi){
    double sph = -3.0/64 * std::sqrt(17/(2 * M_PI) ) * sin(theta) * (715* std::pow(cos(theta) ,7) - 1001* std::pow(cos(theta) ,5) + 385* std::pow(cos(theta) ,3) - 35* cos(theta));
    return {sph * cos(phi), sph * sin(phi)};
}
std::array<double, 2> Y_8_2(double theta, double phi){
    double sph = 3.0/128 * std::sqrt(595/(M_PI) ) * std::pow(sin(theta), 2) * (143* std::pow(cos(theta) ,6) - 143* std::pow(cos(theta) ,4) + 33* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(2 * phi), sph * sin(2 * phi)};
}
std::array<double, 2> Y_8_3(double theta, double phi){
    double sph = -1.0/64 * std::sqrt(19635/(2 * M_PI) ) * std::pow(sin(theta), 3) * (39* std::pow(cos(theta) ,5) - 26* std::pow(cos(theta) ,3) + 3* cos(theta));
    return {sph * cos(3 * phi), sph * sin(3 * phi)};
}
std::array<double, 2> Y_8_4(double theta, double phi){
    double sph = 3.0/128 * std::sqrt(1309/(2 * M_PI) ) * std::pow(sin(theta), 4) * (65* std::pow(cos(theta) ,4) - 26* std::pow(cos(theta) ,2) + 1);
    return {sph * cos(4 * phi), sph * sin(4 * phi)};
}
std::array<double, 2> Y_8_5(double theta, double phi){
    double sph = -3.0/64 * std::sqrt(17017/(2 * M_PI) ) * std::pow(sin(theta), 5) * (5* std::pow(cos(theta) ,3) - cos(theta));
    return {sph * cos(5 * phi), sph * sin(5 * phi)};
}
std::array<double, 2> Y_8_6(double theta, double phi){
    double sph = 1.0/128 * std::sqrt(7293/(M_PI) ) * std::pow(sin(theta), 6) * (15* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(6 * phi), sph * sin(6 * phi)};
}
std::array<double, 2> Y_8_7(double theta, double phi){
    double sph = -3.0/64 * std::sqrt(12155/(2 * M_PI) ) * std::pow(sin(theta), 7) * cos(theta);
    return {sph * cos(7 * phi), sph * sin(7 * phi)};
}
std::array<double, 2> Y_8_8(double theta, double phi){
    double sph = 3.0/256 * std::sqrt(12155/(2 * M_PI) ) * std::pow(sin(theta), 8);
    return {sph * cos(8 * phi), sph * sin(8 * phi)};
}



std::array<double, 2> Y_9_neg9(double theta, double phi){
    double sph = 1.0/512 * std::sqrt(230945/(M_PI) ) * std::pow(sin(theta), 9);
    return {sph * cos(-9 * phi), sph * sin(-9 * phi)};
}
std::array<double, 2> Y_9_neg8(double theta, double phi){
    double sph = 3.0/256 * std::sqrt(230945/(2 * M_PI) ) * std::pow(sin(theta), 8) * cos(theta);
    return {sph * cos(-8 * phi), sph * sin(-8 * phi)};
}
std::array<double, 2> Y_9_neg7(double theta, double phi){
    double sph = 3.0/512 * std::sqrt(13585/(M_PI) ) * std::pow(sin(theta), 7) * (17* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(-7 * phi), sph * sin(-7 * phi)};
}
std::array<double, 2> Y_9_neg6(double theta, double phi){
    double sph = 1.0/128 * std::sqrt(40755/(M_PI) ) * std::pow(sin(theta), 6) * (17* std::pow(cos(theta) ,3) - 3* cos(theta));
    return {sph * cos(-6 * phi), sph * sin(-6 * phi)};
}
std::array<double, 2> Y_9_neg5(double theta, double phi){
    double sph = 3.0/256 * std::sqrt(2717/(M_PI) ) * std::pow(sin(theta), 5) * (85* std::pow(cos(theta) ,4) - 30* std::pow(cos(theta) ,2) + 1);
    return {sph * cos(-5 * phi), sph * sin(-5 * phi)};
}
std::array<double, 2> Y_9_neg4(double theta, double phi){
    double sph = 3.0/128 * std::sqrt(95095/(2 * M_PI) ) * std::pow(sin(theta), 4) * (17* std::pow(cos(theta) ,5) - 10* std::pow(cos(theta) ,3) + cos(theta));
    return {sph * cos(-4 * phi), sph * sin(-4 * phi)};
}
std::array<double, 2> Y_9_neg3(double theta, double phi){
    double sph = 1.0/256 * std::sqrt(21945/(M_PI) ) * std::pow(sin(theta), 3) * (221* std::pow(cos(theta) ,6) - 195* std::pow(cos(theta) ,4) + 39* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(-3 * phi), sph * sin(-3 * phi)};
}
std::array<double, 2> Y_9_neg2(double theta, double phi){
    double sph = 3.0/128 * std::sqrt(1045/(M_PI) ) * std::pow(sin(theta), 2) * (221* std::pow(cos(theta) ,7) - 273* std::pow(cos(theta) ,5) + 91* std::pow(cos(theta) ,3) - 7* cos(theta));
    return {sph * cos(-2 * phi), sph * sin(-2 * phi)};
}
std::array<double, 2> Y_9_neg1(double theta, double phi){
    double sph = 3.0/256 * std::sqrt(95/(2 * M_PI) ) * sin(theta) * (2431* std::pow(cos(theta) ,8) - 4004* std::pow(cos(theta) ,6) + 2002* std::pow(cos(theta) ,4) - 308* std::pow(cos(theta) ,2) + 7);
    return {sph * cos(-phi), sph * sin(-phi)};
}
std::array<double, 2> Y_9_0(double theta, double phi){
    double sph = 1.0/256 * std::sqrt(19/(M_PI) ) * (12155* std::pow(cos(theta) ,9) - 25740* std::pow(cos(theta) ,7) + 18018* std::pow(cos(theta) ,5) - 4620* std::pow(cos(theta) ,3) + 315* cos(theta));
    return {sph * 1.0, 0.0};
}
std::array<double, 2> Y_9_1(double theta, double phi){
    double sph = -3.0/256 * std::sqrt(95/(2 * M_PI) ) * sin(theta) * (2431* std::pow(cos(theta) ,8) - 4004* std::pow(cos(theta) ,6) + 2002* std::pow(cos(theta) ,4) - 308* std::pow(cos(theta) ,2) + 7);
    return {sph * cos(phi), sph * sin(phi)};
}
std::array<double, 2> Y_9_2(double theta, double phi){
    double sph = 3.0/128 * std::sqrt(1045/(M_PI) ) * std::pow(sin(theta), 2) * (221* std::pow(cos(theta) ,7) - 273* std::pow(cos(theta) ,5) + 91* std::pow(cos(theta) ,3) - 7* cos(theta));
    return {sph * cos(2 * phi), sph * sin(2 * phi)};
}
std::array<double, 2> Y_9_3(double theta, double phi){
    double sph = -1.0/256 * std::sqrt(21945/(M_PI) ) * std::pow(sin(theta), 3) * (221* std::pow(cos(theta) ,6) - 195* std::pow(cos(theta) ,4) + 39* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(3 * phi), sph * sin(3 * phi)};
}
std::array<double, 2> Y_9_4(double theta, double phi){
    double sph = 3.0/128 * std::sqrt(95095/(2 * M_PI) ) * std::pow(sin(theta), 4) * (17* std::pow(cos(theta) ,5) - 10* std::pow(cos(theta) ,3) + cos(theta));
    return {sph * cos(4 * phi), sph * sin(4 * phi)};
}
std::array<double, 2> Y_9_5(double theta, double phi){
    double sph = -3.0/256 * std::sqrt(2717/(M_PI) ) * std::pow(sin(theta), 5) * (85* std::pow(cos(theta) ,4) - 30* std::pow(cos(theta) ,2) + 1);
    return {sph * cos(5 * phi), sph * sin(5 * phi)};
}
std::array<double, 2> Y_9_6(double theta, double phi){
    double sph = 1.0/128 * std::sqrt(40755/(M_PI) ) * std::pow(sin(theta), 6) * (17* std::pow(cos(theta) ,3) - 3* cos(theta));
    return {sph * cos(6 * phi), sph * sin(6 * phi)};
}
std::array<double, 2> Y_9_7(double theta, double phi){
    double sph = -3.0/512 * std::sqrt(13585/(M_PI) ) * std::pow(sin(theta), 7) * (17* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(7 * phi), sph * sin(7 * phi)};
}
std::array<double, 2> Y_9_8(double theta, double phi){
    double sph = 3.0/256 * std::sqrt(230945/(2 * M_PI) ) * std::pow(sin(theta), 8) * cos(theta);
    return {sph * cos(8 * phi), sph * sin(8 * phi)};
}
std::array<double, 2> Y_9_9(double theta, double phi){
    double sph = -1.0/512 * std::sqrt(230945/(M_PI) ) * std::pow(sin(theta), 9);
    return {sph * cos(9 * phi), sph * sin(9 * phi)};
}



std::array<double, 2> Y_10_neg10(double theta, double phi){
    double sph = 1.0/1024 * std::sqrt(969969/(M_PI) ) * std::pow(sin(theta), 10);
    return {sph * cos(-10 * phi), sph * sin(-10 * phi)};
}
std::array<double, 2> Y_10_neg9(double theta, double phi){
    double sph = 1.0/512 * std::sqrt(4849845/(M_PI) ) * std::pow(sin(theta), 9) * cos(theta);
    return {sph * cos(-9 * phi), sph * sin(-9 * phi)};
}
std::array<double, 2> Y_10_neg8(double theta, double phi){
    double sph = 1.0/512 * std::sqrt(255255/(2 * M_PI) ) * std::pow(sin(theta), 8) * (19* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(-8 * phi), sph * sin(-8 * phi)};
}
std::array<double, 2> Y_10_neg7(double theta, double phi){
    double sph = 3.0/512 * std::sqrt(85085/(M_PI) ) * std::pow(sin(theta), 7) * (19* std::pow(cos(theta) ,3) - 3* cos(theta));
    return {sph * cos(-7 * phi), sph * sin(-7 * phi)};
}
std::array<double, 2> Y_10_neg6(double theta, double phi){
    double sph = 3.0/1024 * std::sqrt(5005/(M_PI) ) * std::pow(sin(theta), 6) * (323* std::pow(cos(theta) ,4) - 102* std::pow(cos(theta) ,2) + 3);
    return {sph * cos(-6 * phi), sph * sin(-6 * phi)};
}
std::array<double, 2> Y_10_neg5(double theta, double phi){
    double sph = 3.0/256 * std::sqrt(1001/(M_PI) ) * std::pow(sin(theta), 5) * (323* std::pow(cos(theta) ,5) - 170* std::pow(cos(theta) ,3) + 15* cos(theta));
    return {sph * cos(-5 * phi), sph * sin(-5 * phi)};
}
std::array<double, 2> Y_10_neg4(double theta, double phi){
    double sph = 3.0/256 * std::sqrt(5005/(2 * M_PI) ) * std::pow(sin(theta), 4) * (323* std::pow(cos(theta) ,6) - 255* std::pow(cos(theta) ,4) + 45* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(-4 * phi), sph * sin(-4 * phi)};
}
std::array<double, 2> Y_10_neg3(double theta, double phi){
    double sph = 3.0/256 * std::sqrt(5005/(M_PI) ) * std::pow(sin(theta), 3) * (323* std::pow(cos(theta) ,7) - 357* std::pow(cos(theta) ,5) + 105* std::pow(cos(theta) ,3) - 7* cos(theta));
    return {sph * cos(-3 * phi), sph * sin(-3 * phi)};
}
std::array<double, 2> Y_10_neg2(double theta, double phi){
    double sph = 3.0/512 * std::sqrt(385/(2 * M_PI) ) * std::pow(sin(theta), 2) * (4199* std::pow(cos(theta) ,8) - 6188* std::pow(cos(theta) ,6) + 2730* std::pow(cos(theta) ,4) - 364* std::pow(cos(theta) ,2) + 7);
    return {sph * cos(-2 * phi), sph * sin(-2 * phi)};
}
std::array<double, 2> Y_10_neg1(double theta, double phi){
    double sph = 1.0/256 * std::sqrt(1155/(2 * M_PI) ) * sin(theta) * (4199* std::pow(cos(theta) ,9) - 7956* std::pow(cos(theta) ,7) + 4914* std::pow(cos(theta) ,5) - 1092* std::pow(cos(theta) ,3) + 63* cos(theta));
    return {sph * cos(-phi), sph * sin(-phi)};
}
std::array<double, 2> Y_10_0(double theta, double phi){
    double sph = 1.0/512 * std::sqrt(21/(M_PI) ) * (46189 * std::pow(cos(theta), 10) - 109395* std::pow(cos(theta) ,8) + 90090* std::pow(cos(theta) ,6) - 30030* std::pow(cos(theta) ,4) + 3465* std::pow(cos(theta) ,2) - 63);
    return {sph * 1.0, 0.0};
}
std::array<double, 2> Y_10_1(double theta, double phi){
    double sph = -1.0/256 * std::sqrt(1155/(2 * M_PI) ) * sin(theta) * (4199* std::pow(cos(theta) ,9) - 7956* std::pow(cos(theta) ,7) + 4914* std::pow(cos(theta) ,5) - 1092* std::pow(cos(theta) ,3) + 63* cos(theta));
    return {sph * cos(phi), sph * sin(phi)};
}
std::array<double, 2> Y_10_2(double theta, double phi){
    double sph = 3.0/512 * std::sqrt(385/(2 * M_PI) ) * std::pow(sin(theta), 2) * (4199* std::pow(cos(theta) ,8) - 6188* std::pow(cos(theta) ,6) + 2730* std::pow(cos(theta) ,4) - 364* std::pow(cos(theta) ,2) + 7);
    return {sph * cos(2 * phi), sph * sin(2 * phi)};
}
std::array<double, 2> Y_10_3(double theta, double phi){
    double sph = -3.0/256 * std::sqrt(5005/(M_PI) ) * std::pow(sin(theta), 3) * (323* std::pow(cos(theta) ,7) - 357* std::pow(cos(theta) ,5) + 105* std::pow(cos(theta) ,3) - 7* cos(theta));
    return {sph * cos(3 * phi), sph * sin(3 * phi)};
}
std::array<double, 2> Y_10_4(double theta, double phi){
    double sph = 3.0/256 * std::sqrt(5005/(2 * M_PI) ) * std::pow(sin(theta), 4) * (323* std::pow(cos(theta) ,6) - 255* std::pow(cos(theta) ,4) + 45* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(4 * phi), sph * sin(4 * phi)};
}
std::array<double, 2> Y_10_5(double theta, double phi){
    double sph = -3.0/256 * std::sqrt(1001/(M_PI) ) * std::pow(sin(theta), 5) * (323* std::pow(cos(theta) ,5) - 170* std::pow(cos(theta) ,3) + 15* cos(theta));
    return {sph * cos(5 * phi), sph * sin(5 * phi)};
}
std::array<double, 2> Y_10_6(double theta, double phi){
    double sph = 3.0/1024 * std::sqrt(5005/(M_PI) ) * std::pow(sin(theta), 6) * (323* std::pow(cos(theta) ,4) - 102* std::pow(cos(theta) ,2) + 3);
    return {sph * cos(6 * phi), sph * sin(6 * phi)};
}
std::array<double, 2> Y_10_7(double theta, double phi){
    double sph = -3.0/512 * std::sqrt(85085/(M_PI) ) * std::pow(sin(theta), 7) * (19* std::pow(cos(theta) ,3) - 3* cos(theta));
    return {sph * cos(7 * phi), sph * sin(7 * phi)};
}
std::array<double, 2> Y_10_8(double theta, double phi){
    double sph = 1.0/512 * std::sqrt(255255/(2 * M_PI) ) * std::pow(sin(theta), 8) * (19* std::pow(cos(theta) ,2) - 1);
    return {sph * cos(8 * phi), sph * sin(8 * phi)};
}
std::array<double, 2> Y_10_9(double theta, double phi){
    double sph = -1.0/512 * std::sqrt(4849845/(M_PI) ) * std::pow(sin(theta), 9) * cos(theta);
    return {sph * cos(9 * phi), sph * sin(9 * phi)};
}
std::array<double, 2> Y_10_10(double theta, double phi){
    double sph = 1.0/1024 * std::sqrt(969969/(M_PI) ) * std::pow(sin(theta), 10);
    return {sph * cos(10 * phi), sph * sin(10 * phi)};
}


std::array<double, 2> Ylmi_analytical(int l,int m, double theta, double phi){
    switch (l) {
        case 0: {
            return Y_0_0(theta, phi);
        }
        case 1: {
            switch(m){
                default: throw std::runtime_error("Invalid value of m.");
                case -1: return Y_1_neg1(theta, phi);
                case 0: return Y_1_0(theta, phi);
                case 1: return Y_1_1(theta, phi);
            }
        }
        case 2: {
            switch(m){
                default: throw std::runtime_error("Invalid value of m.");
                case -2: return Y_2_neg2(theta, phi);
                case -1: return Y_2_neg1(theta, phi);
                case 0: return Y_2_0(theta, phi);
                case 1: return Y_2_1(theta, phi);
                case 2: return Y_2_2(theta, phi);
            }
        }
        case 3: {
            switch(m){
                default: throw std::runtime_error("Invalid value of m.");
                case -3:return Y_3_neg3(theta, phi);
                case -2:return Y_3_neg2(theta, phi);
                case -1:return Y_3_neg1(theta, phi);
                case 0:return Y_3_0(theta, phi);
                case 1:return Y_3_1(theta, phi);
                case 2:return Y_3_2(theta, phi);
                case 3:return Y_3_3(theta, phi);
            }
        }
        case 4: {
            switch(m){
                default: throw std::runtime_error("Invalid value of m.");
                case -4: return Y_4_neg4(theta, phi);
                case -3: return Y_4_neg3(theta, phi);
                case -2: return Y_4_neg2(theta, phi);
                case -1: return Y_4_neg1(theta, phi);
                case 0: return Y_4_0(theta, phi);
                case 1: return Y_4_1(theta, phi);
                case 2: return Y_4_2(theta, phi);
                case 3: return Y_4_3(theta, phi);
                case 4: return Y_4_4(theta, phi);
            }
        }
        case 5: {
            switch(m){
                default: throw std::runtime_error("Invalid value of m.");
                case -5: return Y_5_neg5(theta, phi);
                case -4: return Y_5_neg4(theta, phi);
                case -3: return Y_5_neg3(theta, phi);
                case -2: return Y_5_neg2(theta, phi);
                case -1: return Y_5_neg1(theta, phi);
                case 0: return Y_5_0(theta, phi);
                case 1: return Y_5_1(theta, phi);
                case 2: return Y_5_2(theta, phi);
                case 3: return Y_5_3(theta, phi);
                case 4: return Y_5_4(theta, phi);
                case 5: return Y_5_5(theta, phi);
            }
        }
        case 6: {
            switch(m){
                default: throw std::runtime_error("Invalid value of m.");
                case -6: return Y_6_neg6(theta, phi);
                case -5: return Y_6_neg5(theta, phi);
                case -4: return Y_6_neg4(theta, phi);
                case -3: return Y_6_neg3(theta, phi);
                case -2: return Y_6_neg2(theta, phi);
                case -1: return Y_6_neg1(theta, phi);
                case 0: return Y_6_0(theta, phi);
                case 1: return Y_6_1(theta, phi);
                case 2: return Y_6_2(theta, phi);
                case 3: return Y_6_3(theta, phi);
                case 4: return Y_6_4(theta, phi);
                case 5: return Y_6_5(theta, phi);
                case 6: return Y_6_6(theta, phi);
            }
        }
        case 7: {
            switch(m){
                default: throw std::runtime_error("Invalid value of m.");
                case -7: return Y_7_neg7(theta, phi);
                case -6: return Y_7_neg6(theta, phi);
                case -5: return Y_7_neg5(theta, phi);
                case -4: return Y_7_neg4(theta, phi);
                case -3: return Y_7_neg3(theta, phi);
                case -2: return Y_7_neg2(theta, phi);
                case -1: return Y_7_neg1(theta, phi);
                case 0: return Y_7_0(theta, phi);
                case 1: return Y_7_1(theta, phi);
                case 2: return Y_7_2(theta, phi);
                case 3: return Y_7_3(theta, phi);
                case 4: return Y_7_4(theta, phi);
                case 5: return Y_7_5(theta, phi);
                case 6: return Y_7_6(theta, phi);
                case 7: return Y_7_7(theta, phi);
            }
        }
        case 8: {
            switch(m){
                default: throw std::runtime_error("Invalid value of m.");
                case -8: return Y_8_neg8(theta, phi);
                case -7: return Y_8_neg7(theta, phi);
                case -6: return Y_8_neg6(theta, phi);
                case -5: return Y_8_neg5(theta, phi);
                case -4: return Y_8_neg4(theta, phi);
                case -3: return Y_8_neg3(theta, phi);
                case -2: return Y_8_neg2(theta, phi);
                case -1: return Y_8_neg1(theta, phi);
                case 0: return Y_8_0(theta, phi);
                case 1: return Y_8_1(theta, phi);
                case 2: return Y_8_2(theta, phi);
                case 3: return Y_8_3(theta, phi);
                case 4: return Y_8_4(theta, phi);
                case 5: return Y_8_5(theta, phi);
                case 6: return Y_8_6(theta, phi);
                case 7: return Y_8_7(theta, phi);
                case 8: return Y_8_8(theta, phi);
            }
        }
        case 9: {
            switch(m){
                default: throw std::runtime_error("Invalid value of m.");
                case -9: return Y_9_neg9(theta, phi);
                case -8: return Y_9_neg8(theta, phi);
                case -7: return Y_9_neg7(theta, phi);
                case -6: return Y_9_neg6(theta, phi);
                case -5: return Y_9_neg5(theta, phi);
                case -4: return Y_9_neg4(theta, phi);
                case -3: return Y_9_neg3(theta, phi);
                case -2: return Y_9_neg2(theta, phi);
                case -1: return Y_9_neg1(theta, phi);
                case 0: return Y_9_0(theta, phi);
                case 1: return Y_9_1(theta, phi);
                case 2: return Y_9_2(theta, phi);
                case 3: return Y_9_3(theta, phi);
                case 4: return Y_9_4(theta, phi);
                case 5: return Y_9_5(theta, phi);
                case 6: return Y_9_6(theta, phi);
                case 7: return Y_9_7(theta, phi);
                case 8: return Y_9_8(theta, phi);
                case 9: return Y_9_9(theta, phi);
            }
        }
        case 10: {
            switch(m){
                default: throw std::runtime_error("Invalid value of m.");
                case -10: return Y_10_neg10(theta, phi);
                case -9: return Y_10_neg9(theta, phi);
                case -8: return Y_10_neg8(theta, phi);
                case -7: return Y_10_neg7(theta, phi);
                case -6: return Y_10_neg6(theta, phi);
                case -5: return Y_10_neg5(theta, phi);
                case -4: return Y_10_neg4(theta, phi);
                case -3: return Y_10_neg3(theta, phi);
                case -2: return Y_10_neg2(theta, phi);
                case -1: return Y_10_neg1(theta, phi);
                case 0: return Y_10_0(theta, phi);
                case 1: return Y_10_1(theta, phi);
                case 2: return Y_10_2(theta, phi);
                case 3: return Y_10_3(theta, phi);
                case 4: return Y_10_4(theta, phi);
                case 5: return Y_10_5(theta, phi);
                case 6: return Y_10_6(theta, phi);
                case 7: return Y_10_7(theta, phi);
                case 8: return Y_10_8(theta, phi);
                case 9: return Y_10_9(theta, phi);
                case 10: return Y_10_10(theta, phi);
            }
        }
        default:{
            throw std::runtime_error("Only l <= 10 have analytical expressions.");
        }
    }
}