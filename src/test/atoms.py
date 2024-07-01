
import reduced_parameter as red

atoms = AtomsInBox(box_length = 20, cell_width = 5, mass_dict = red.mass_dict)

atoms.random_initialize({"Ar":5,"Xe":5}, max_velocity = 10)

idxAr = atoms.element_idx(element = "Xe")
idxXe = atoms.element_idx(element = "Ar")

epsilon_mat, sigma_mat = red.generate_LJparam_matrix(idxAr = idxAr, idxXe = idxXe)

atoms.add_calculator(calculator_kwargs = {
    "epsilon": epsilon_mat, "sigma" : sigma_mat, 
    "positive_atom_idx" : idxXe, "negative_atom_idx" : idxAr,
    "mu0" : red.mu0, "d" : red.d0, "a" : red.a
    })

gradD = atoms.dipole_grad()
print(gradD.shape)

