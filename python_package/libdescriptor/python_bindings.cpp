#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <iostream>
#include "Descriptors.hpp"
#include <cstdlib>
#include <memory>

#include "neighbor/neighbor_list.h"

#define MY_WARNING(message)                                           \
  {                                                                   \
    std::cerr << "* Error (Neighbor List) : \"" << message            \
              << "\" : " << __LINE__ << ":" << __FILE__ << std::endl; \
  }


namespace py = pybind11;

namespace {
    struct PyNeighListDestroy {
        void operator()(NeighList *neighList) const { nbl_clean(&neighList); }
    };
}  // namespace


using namespace Descriptor;

// Trampoline class for DescriptorKind pure virtual compute function
class PyDescriptorKind : public DescriptorKind {
public:
    using DescriptorKind::DescriptorKind;

    void compute(int index,
                 int n_atoms,
                 int *species,
                 int *neighbor_lists,
                 int number_of_neighbors,
                 double *coordinates,
                 double *desc) override {
        PYBIND11_OVERRIDE_PURE(void, DescriptorKind, compute,
                               index, n_atoms, species, neighbor_lists, number_of_neighbors, coordinates, desc
        );
    };
};


PYBIND11_MODULE(libdescriptor, m) {
    m.doc() = "Python interface to libdescriptor";

    // Python bindings for AvailableDescriptor kinds.
    py::enum_<AvailableDescriptor>(m, "AvailableDescriptors")
            .value("Bispectrum", AvailableDescriptor::KindBispectrum)
            .value("SymmetryFunctions", AvailableDescriptor::KindSymmetryFunctions)
            .value("SOAP", AvailableDescriptor::KindSOAP);
            //.value("Xi", AvailableDescriptor::KindXi);

    // Python bindings for DescriptorKind
    py::class_<DescriptorKind, PyDescriptorKind>(m, "DescriptorKind")
            .def(py::init<>()) // Default constructor

                    // Return default constructor, mostly needed for creating empty objects, may need in future for
                    // hyperparameter optimization
            .def("init_descriptor", py::overload_cast<AvailableDescriptor>(&DescriptorKind::initDescriptor))

                    // Return descriptor object created from reading the input file
            .def("init_descriptor",
                 py::overload_cast<std::string &, AvailableDescriptor>(&DescriptorKind::initDescriptor))

                    // Return descriptor object created from Symmetry Functions, using direct parameters
            .def("init_descriptor", py::overload_cast<AvailableDescriptor,
                    std::vector<std::string> *, std::string *, double *, std::vector<std::string> *, std::vector<int> *,
                    std::vector<double> *>(&DescriptorKind::initDescriptor))

                    // Return descriptor object created from Bispectrum, using direct parameters
            .def("init_descriptor", py::overload_cast<AvailableDescriptor, double, int, int, int, double, int,
                    int, double *, std::vector<std::string> *, std::vector<double> *>(&DescriptorKind::initDescriptor))

          //          // Return descriptor object created from Xi, using direct parameters
          //  .def("init_descriptor", py::overload_cast<AvailableDescriptor, int, double,
          //          std::vector<std::string> &, std::string &>(&DescriptorKind::initDescriptor))

                    // Return descriptor object created from SOAP, using direct parameters
            .def("init_descriptor", py::overload_cast<AvailableDescriptor, int, int, double,
                    std::vector<std::string> &, std::string, double>(&DescriptorKind::initDescriptor))

                    // Compute function for calculating the descriptor
            .def("compute",
                 [](DescriptorKind &ds,
                    int index,
                    py::array_t<int, py::array::c_style | py::array::forcecast> &species,
                    py::array_t<int, py::array::c_style | py::array::forcecast> &neighbors,
                    py::array_t<double, py::array::c_style | py::array::forcecast> &coords) {
                     int n_atoms = static_cast<int>(coords.shape(0));
                     int n_neigh = static_cast<int>(neighbors.shape(0));
                     auto desc = new double[ds.width];
                     for (int i = 0; i < ds.width; i++) desc[i] = 0.0;
                     ds.compute(index,
                                n_atoms,
                                const_cast<int *>(species.data(0)),
                                const_cast<int *>(neighbors.data(0)),
                                n_neigh,
                                const_cast<double *>(coords.data(0)),
                                desc);

                     py::array_t<double> desc_array(ds.width, desc);
                     return desc_array;
                 }, py::return_value_policy::take_ownership)
            .def_readwrite("param_file", &DescriptorKind::descriptor_param_file)
            .def_readwrite("kind", &DescriptorKind::descriptor_kind)
            .def_readwrite("width", &DescriptorKind::width);


    // Calculate descriptor for a single atom using the descriptor supplied in DescriptorKind argument
    m.def("compute_single_atom",
          [](DescriptorKind &ds, int index, py::array_t<int, py::array::c_style | py::array::forcecast> &species,
             py::array_t<int, py::array::c_style | py::array::forcecast> &neighbors,
             py::array_t<double, py::array::c_style | py::array::forcecast> &coordinates) {
              int n_atoms = static_cast<int>(coordinates.shape(0));
              int n_neigh = static_cast<int>(neighbors.shape(0));
              auto desc = new double[ds.width];
              for (int i = 0; i < ds.width; i++) desc[i] = 0.0;
              compute_single_atom(index,
                                  n_atoms,
                                  const_cast<int *>(species.data(0)),
                                  const_cast<int *>(neighbors.data(0)),
                                  n_neigh,
                                  const_cast<double *>(coordinates.data(0)),
                                  desc,
                                  &ds);

              py::array_t<double> desc_array(ds.width, desc);
              return desc_array;
          }, py::return_value_policy::take_ownership,
          "Compute descriptor for single atom of configuration.");

    m.def("compute",
          [](DescriptorKind &ds,
             int n_atoms,
             py::array_t<int, py::array::c_style | py::array::forcecast> &species,
             py::array_t<int, py::array::c_style | py::array::forcecast> &neighbors,
             py::array_t<int, py::array::c_style | py::array::forcecast> &number_of_neighbors,
             py::array_t<double, py::array::c_style | py::array::forcecast> &coordinates) {
              auto desc = new double[ds.width * n_atoms];
              for (int i = 0; i < ds.width * n_atoms; i++) desc[i] = 0.0;
              compute(n_atoms,
                      const_cast<int *>(species.data(0)),
                      const_cast<int *>(neighbors.data(0)),
                      const_cast<int *>(number_of_neighbors.data(0)),
                      const_cast<double *>(coordinates.data(0)),
                      desc,
                      &ds);

              py::array_t<double> desc_array({n_atoms, ds.width}, desc);
              return desc_array;
          }, py::return_value_policy::take_ownership,
          "Compute descriptor for single atom of configuration.");

    m.def("compute_batch",
          [](DescriptorKind &ds,
             py::array_t<int, py::array::c_style | py::array::forcecast> &n_atoms,
             py::array_t<int, py::array::c_style | py::array::forcecast> &config_ptr,
             py::array_t<int, py::array::c_style | py::array::forcecast> &species,
             py::array_t<int, py::array::c_style | py::array::forcecast> &neighbors,
             py::array_t<int, py::array::c_style | py::array::forcecast> &number_of_neighbors,
             py::array_t<double, py::array::c_style | py::array::forcecast> &coordinates
          ) {
              int n_configurations = static_cast<int>(n_atoms.size());
              int n_total_atoms = 0;
              for (int i = 0; i < n_configurations; i++) {
                  n_total_atoms += n_atoms.at(i);
              }
              auto desc = new double[ds.width * n_total_atoms];
              std::fill(desc, desc + ds.width * n_total_atoms, 0.0);

              compute_batch(
                      n_configurations,
                      const_cast<int *>(n_atoms.data(0)),
                      const_cast<int *>(config_ptr.data(0)),
                      const_cast<int *>(species.data(0)),
                      const_cast<int *>(neighbors.data(0)),
                      const_cast<int *>(number_of_neighbors.data(0)),
                      const_cast<double *>(coordinates.data(0)),
                      desc,
                      &ds);

              py::array_t<double> desc_array({n_total_atoms, ds.width}, desc);
              return desc_array;
          }, py::return_value_policy::take_ownership,
          "Compute descriptor for multiple configurations.");

    // Calculate gradient of descriptor for a single atom using the descriptor supplied in DescriptorKind argument
    m.def("gradient_single_atom",
          [](DescriptorKind &ds, int index, py::array_t<int, py::array::c_style | py::array::forcecast> &species,
             py::array_t<int, py::array::c_style | py::array::forcecast> &neighbors,
             py::array_t<double, py::array::c_style | py::array::forcecast> &coordinates,
             py::array_t<double, py::array::c_style | py::array::forcecast> &desc,
             py::array_t<double, py::array::c_style | py::array::forcecast> &dE_ddesc) {
              int n_atoms = static_cast<int>(coordinates.shape(0));
              int n_neigh = static_cast<int>(neighbors.shape(0));
              auto d_coordinates = new double[coordinates.size()];
              auto desc_throwaway = new double[ds.width]; // TODO: use smart pointer
              for (int i = 0; i < coordinates.size(); i++) d_coordinates[i] = 0.0;

              std::copy(desc.data(0), desc.data(0) + ds.width, desc_throwaway);

              gradient_single_atom(index,
                                   n_atoms,
                                   const_cast<int *>(species.data(0)),
                                   const_cast<int *>(neighbors.data(0)),
                                   n_neigh,
                                   const_cast<double *>(coordinates.data(0)),
                                   d_coordinates,
                                   desc_throwaway,
                                   const_cast<double *>(dE_ddesc.data(0)),
                                   &ds);
              delete[] desc_throwaway;
              py::array_t<double> d_coord_array({coordinates.shape(0), coordinates.shape(1)}, d_coordinates);
              return d_coord_array;
          }, py::return_value_policy::take_ownership,
          "Compute gradient of descriptor for single atom configuration.");
    // TODO generate_one_atom, which is compatible with current kliff

    m.def("gradient",
          [](DescriptorKind &ds,
             int n_atoms,
             py::array_t<int, py::array::c_style | py::array::forcecast> &species,
             py::array_t<int, py::array::c_style | py::array::forcecast> &neighbors,
             py::array_t<int, py::array::c_style | py::array::forcecast> &number_of_neighbors,
             py::array_t<double, py::array::c_style | py::array::forcecast> &coordinates,
             py::array_t<double, py::array::c_style | py::array::forcecast> &desc,
             py::array_t<double, py::array::c_style | py::array::forcecast> &dE_ddesc) {
              auto d_coordinates = new double[coordinates.size()];
              auto desc_throwaway = new double[ds.width * n_atoms]; // TODO: use smart pointer
              // copy desc to desc_throwaway, enzyme will mutate desc
              std::copy(desc.data(0), desc.data(0) + ds.width * n_atoms, desc_throwaway);


              for (int i = 0; i < coordinates.size(); i++) d_coordinates[i] = 0.0;
              gradient(n_atoms,
                       const_cast<int *>(species.data(0)),
                       const_cast<int *>(neighbors.data(0)),
                       const_cast<int *>(number_of_neighbors.data(0)),
                       const_cast<double *>(coordinates.data(0)),
                       d_coordinates,
                       desc_throwaway,
                       const_cast<double *>(dE_ddesc.data(0)),
                       &ds);
              delete[] desc_throwaway;
              py::array_t<double> d_coord_array({coordinates.shape(0), coordinates.shape(1)}, d_coordinates);
              return d_coord_array;
          }, py::return_value_policy::take_ownership,
          "Compute gradient of descriptor for complete configuration.");

    m.def("gradient_batch", [](DescriptorKind &ds,
                               py::array_t<int, py::array::c_style | py::array::forcecast> &n_atoms,
                               py::array_t<int, py::array::c_style | py::array::forcecast> &config_ptr,
                               py::array_t<int, py::array::c_style | py::array::forcecast> &species,
                               py::array_t<int, py::array::c_style | py::array::forcecast> &neighbors,
                               py::array_t<int, py::array::c_style | py::array::forcecast> &number_of_neighbors,
                               py::array_t<double, py::array::c_style | py::array::forcecast> &coordinates,
                               py::array_t<double, py::array::c_style | py::array::forcecast> &desc,
                               py::array_t<double, py::array::c_style | py::array::forcecast> &dE_ddesc
    ) {
        int n_configurations = static_cast<int>(n_atoms.size());
        int n_total_atoms = 0;
        for (int i = 0; i < n_configurations; i++) {
            n_total_atoms += n_atoms.at(i);
        }
        auto d_coordinates = new double[coordinates.size()];
        auto desc_throwaway = new double[ds.width * n_total_atoms]; // TODO: use smart pointer
        // copy desc to desc_throwaway, enzyme will mutate desc
        std::copy(desc.data(0), desc.data(0) + ds.width * n_total_atoms, desc_throwaway);
        std::fill(d_coordinates, d_coordinates + coordinates.size(), 0.0);
        gradient_batch(
                n_configurations,
                const_cast<int *>(n_atoms.data(0)),
                const_cast<int *>(config_ptr.data(0)),
                const_cast<int *>(species.data(0)),
                const_cast<int *>(neighbors.data(0)),
                const_cast<int *>(number_of_neighbors.data(0)),
                const_cast<double *>(coordinates.data(0)),
                d_coordinates,
                desc_throwaway,
                const_cast<double *>(dE_ddesc.data(0)),
                &ds);

        delete[] desc_throwaway;
        py::array_t<double> d_coord_array({coordinates.shape(0), coordinates.shape(1)}, d_coordinates);
        return d_coord_array;
    }, "Compute gradient of descriptor for multiple configurations.");

    m.def("jacobian",
          [](DescriptorKind &ds,
             int n_atoms,
             py::array_t<int, py::array::c_style | py::array::forcecast> &species,
             py::array_t<int, py::array::c_style | py::array::forcecast> &neighbors,
             py::array_t<int, py::array::c_style | py::array::forcecast> &number_of_neighbors,
             py::array_t<double, py::array::c_style | py::array::forcecast> &coordinates) {
              int n_total_atoms = static_cast<int>(coordinates.shape(0));
              auto J_coordinates = new double[coordinates.size() * ds.width * n_atoms];
              std::fill(J_coordinates, J_coordinates + coordinates.size() * ds.width * n_atoms, 0);
              jacobian(n_atoms,
                       n_total_atoms,
                       const_cast<int *>(species.data(0)),
                       const_cast<int *>(neighbors.data(0)),
                       const_cast<int *>(number_of_neighbors.data(0)),
                       const_cast<double *>(coordinates.data(0)),
                       J_coordinates,
                       &ds);
              py::array_t<double> J_coord_array({ds.width * n_atoms, static_cast<int>(coordinates.size())},
                                                J_coordinates);
              return J_coord_array;
          }, py::return_value_policy::take_ownership,
          "Compute Jacobian of descriptor for complete configuration");

    // Calculate gradient for a single atom using the numerical differentiation
//    m.def("num_gradient_single_atom",
//          [](DescriptorKind &ds, int index, py::array_t<int, py::array::c_style | py::array::forcecast> &species,
//             py::array_t<int, py::array::c_style | py::array::forcecast> &neighbors,
//             py::array_t<double, py::array::c_style | py::array::forcecast> &coordinates,
//             py::array_t<double, py::array::c_style | py::array::forcecast> &dE_ddesc) {
//              int n_atoms = static_cast<int>(coordinates.shape(0));
//              int n_neigh = static_cast<int>(neighbors.shape(0));
//              auto d_coordinates = new double[3];
//              for (int i = 0; i < 3; i++) d_coordinates[i] = 0.0;
//              num_gradient_single_atom(index,
//                                       n_atoms,
//                                       const_cast<int *>(species.data(0)),
//                                       const_cast<int *>(neighbors.data(0)),
//                                       n_neigh,
//                                       const_cast<double *>(coordinates.data(0)),
//                                       d_coordinates,
//                                       const_cast<double *>(dE_ddesc.data(0)),
//                                       &ds);
//              py::array_t<double> d_coord_array(3, d_coordinates);
//              return d_coord_array;
//          }, py::return_value_policy::take_ownership,
//          "Compute gradient of descriptor for single atom configuration.");


    /******************************************************************************
     *  Neighborlist for minimal usability
     * borrowed from KLIFF
     *****************************************************************************/
    py::class_<NeighList, std::unique_ptr<NeighList, PyNeighListDestroy> >(
            m, "NeighList", py::module_local())
            .def(py::init([]() {
                NeighList *neighList = new NeighList;
                return std::unique_ptr<NeighList, PyNeighListDestroy>(std::move(neighList));
            }))
            .def("build",
                 [](NeighList &self,
                    py::array_t<double> coords,
                    double const influence_distance,
                    py::array_t<double> cutoffs,
                    py::array_t<int> need_neigh) {
                     int const natoms_1 = static_cast<int>(coords.size() / 3);
                     int const natoms_2 = static_cast<int>(need_neigh.size());

                     if (natoms_1 != natoms_2) {
                         MY_WARNING("\"coords\" size and \"need_neigh\" size do not match!");
                     }

                     int const natoms = natoms_1 <= natoms_2 ? natoms_1 : natoms_2;
                     double const *coords_data = coords.data();
                     int const number_of_cutoffs = static_cast<int>(cutoffs.size());
                     double const *cutoffs_data = cutoffs.data();
                     int const *need_neigh_data = need_neigh.data();

                     int error = nbl_build(&self,
                                           natoms,
                                           coords_data,
                                           influence_distance,
                                           number_of_cutoffs,
                                           cutoffs_data,
                                           need_neigh_data);
                     if (error == 1) {
                         throw std::runtime_error("Cell size too large! (partilces fly away) or\n"
                                                  "Collision of atoms happened!");
                     }
                 }, "Build the neighbor list.",
                 py::arg("coords").noconvert(),
                 py::arg("influence_distance"),
                 py::arg("cutoffs").noconvert(),
                 py::arg("need_neigh").noconvert())
            .def("get_neigh",
                 [](NeighList &self,
                    py::array_t<double> cutoffs,
                    int const neighbor_list_index,
                    int const particle_number) {
                     int const number_of_cutoffs = static_cast<int>(cutoffs.size());
                     double const *cutoffs_data = cutoffs.data();
                     int number_of_neighbors = 0;
                     int const *neigh_of_atom;

                     void const *const data_object
                             = reinterpret_cast<void const *const>(&self);

                     int error = nbl_get_neigh(data_object,
                                               number_of_cutoffs,
                                               cutoffs_data,
                                               neighbor_list_index,
                                               particle_number,
                                               &number_of_neighbors,
                                               &neigh_of_atom);
                     if (error == 1) {
                         if (neighbor_list_index >= self.numberOfNeighborLists) {
                             throw std::runtime_error("neighbor_list_index = "
                                                      + std::to_string(neighbor_list_index)
                                                      + " >= self.numberOfNeighborLists = "
                                                      + std::to_string(self.numberOfNeighborLists));
                         } else if (cutoffs_data[neighbor_list_index]
                                    > self.lists[neighbor_list_index].cutoff) {
                             throw std::runtime_error(
                                     "cutoffs_data[neighbor_list_index] = "
                                     + std::to_string(cutoffs_data[neighbor_list_index])
                                     + " > self.lists[neighbor_list_index].cutoff = "
                                     + std::to_string(self.lists[neighbor_list_index].cutoff));
                         } else {
                             throw std::runtime_error(
                                     "particle_number = " + std::to_string(particle_number) + " < 0!");
                         }
                     }

                     // pack as a numpy array
                     auto neighbors_of_particle = py::array(py::buffer_info(
                             const_cast<int *>(neigh_of_atom),  // data pointer
                             sizeof(int),  // size of one element
                             py::format_descriptor<int>::format(),  // Python struct-style
                             // format descriptor
                             1,  // dimension
                             {number_of_neighbors},  // size of each dimension
                             {sizeof(int)}  // stride of each dimension
                     ));

                     py::tuple re(2);
                     re[0] = number_of_neighbors;
                     re[1] = neighbors_of_particle;
                     return re;
                 }, R"pbdoc(
     Get the number of neighbors and neighbors of particle.

     Returns:
         int, 1darray: number_of_neighbors, neighbors_of_particle
     )pbdoc",
                 py::arg("cutoffs").noconvert(),
                 py::arg("neighbor_list_index"),
                 py::arg("particle_number"));

    m.def("create", []() {
              NeighList *neighList = new NeighList;
              return std::unique_ptr<NeighList, PyNeighListDestroy>(std::move(neighList));
          }, R"pbdoc(
     Create a new NeighList object.

     Returns:
         NeighList: neighList
     )pbdoc"
    );

    // cannot bind `nbl_get_neigh_kim` directly, since it has pointer arguments
    // so we return a pointer to this function
    m.def("get_neigh_kim", []() {
        // the allowed return pointer type by pybind11 is: void const *
        // so cast the function pointer to it, and we need to cast back when
        // using it
        return (void const *) &nbl_get_neigh;
    });

    m.def("create_paddings",
          [](double const influence_distance,
             py::array_t<double> cell,
             py::array_t<int> pbc,
             py::array_t<double> coords,
             py::array_t<int> species) {
              int const natoms_1 = static_cast<int>(coords.size() / 3);
              int const natoms_2 = static_cast<int>(species.size());

              if (natoms_1 != natoms_2) {
                  MY_WARNING("\"coords\" size and \"need_neigh\" size do not match!");
              }

              int const natoms = natoms_1 <= natoms_2 ? natoms_1 : natoms_2;
              double const *cell_data = cell.data();
              int const *pbc_data = pbc.data();
              double const *coords_data = coords.data();
              int const *species_data = species.data();

              int number_of_pads;
              std::vector<double> pad_coords;
              std::vector<int> pad_species;
              std::vector<int> pad_image;

              int error = nbl_create_paddings(natoms,
                                              influence_distance,
                                              cell_data,
                                              pbc_data,
                                              coords_data,
                                              species_data,
                                              number_of_pads,
                                              pad_coords,
                                              pad_species,
                                              pad_image);
              if (error == 1) {
                  throw std::runtime_error(
                          "In inverting the cell matrix, the determinant is 0!");
              }

              // pack as a 2D numpy array
              auto coordinates_of_paddings
                      = py::array(py::buffer_info(pad_coords.data(),
                                                  sizeof(double),
                                                  py::format_descriptor<double>::format(),
                                                  2,
                                                  {number_of_pads, 3},
                                                  {sizeof(double) * 3, sizeof(double)}));

              // pack as a numpy array
              auto species_code_of_paddings
                      = py::array(py::buffer_info(pad_species.data(),
                                                  sizeof(int),
                                                  py::format_descriptor<int>::format(),
                                                  1,
                                                  {number_of_pads},
                                                  {sizeof(int)}));

              // pack as a numpy array
              auto master_particle_of_paddings
                      = py::array(py::buffer_info(pad_image.data(),
                                                  sizeof(int),
                                                  py::format_descriptor<int>::format(),
                                                  1,
                                                  {number_of_pads},
                                                  {sizeof(int)}));

              py::tuple re(3);
              re[0] = coordinates_of_paddings;
              re[1] = species_code_of_paddings;
              re[2] = master_particle_of_paddings;
              return re;
          }, R"pbdoc(
     Create padding.

     Returns:
         2darray, 1darray, 1darray: coordinates_of_paddings,
             species_code_of_paddings, master_particle_of_paddings
     )pbdoc",
          py::arg("influence_distance"),
          py::arg("cell").noconvert(),
          py::arg("pbc").noconvert(),
          py::arg("coords").noconvert(),
          py::arg("species").noconvert());
}
