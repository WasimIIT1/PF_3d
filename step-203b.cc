/*Shear test in 2d with splitting and load displacement curves: Phasefield*/


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <iostream>
#include <deal.II/fe/component_mask.h>
#include <deal.II/lac/sparse_direct.h>
#include <algorithm>

namespace step203
{
using namespace dealii;




class PhaseField
{
	// Function declarations
public:
	PhaseField();
	void run();
	void elastic_solver(int t);
	void damage_solver();

private:

	void setup_boundary_values_elastic();
	void setup_system_elastic();
	void assemble_system_elastic(int t);
	void solve_elastic();
	void output_results_elastic( int t) const;

	void setup_boundary_values_damage();
	void setup_system_damage();
	void assemble_system_damage();
	void solve_damage();
	void output_results( int t) const;

	void output_results_damage( int t) const;
	float H_plus(Vector<double> & solution_elastic, const auto cell,const unsigned int q_point,const FEValues<2> & fe_values_damage);
	float damage_gauss_pt(Vector<double> & solution_damage,const auto cell
			,const unsigned int q_point,const FEValues<2> & fe_values_elastic
	);


	//deallog.depth_console(2);

	// Creating objects for elasticity
	Triangulation<2> triangulation;
	FESystem<2> fe_elastic;
	DoFHandler<2>    dof_handler_elastic;

	std::map<types::global_dof_index, double> boundary_values_elastic;
	SparsityPattern      sparsity_pattern_elastic;
	SparseMatrix<double> system_matrix_elastic;
	Vector<double> solution_elastic;
	Vector<double> system_rhs_elastic;

	// Creating objects for damage
	FE_Q<2>          fe_damage;
	DoFHandler<2>    dof_handler_damage;

	std::map<types::global_dof_index, double> boundary_values_damage;
	SparsityPattern      sparsity_pattern_damage;
	SparseMatrix<double> system_matrix_damage;
	Vector<double> solution_damage;
	Vector<double> system_rhs_damage;
	Vector<double> solution_damage_old;
	Vector<double> solution_elastic_old;

	float error_elastic_solution;
	float error_elastic_solution_numerator;
	float error_elastic_solution_denominator;
	float error_damage_solution;
	float error_damage_solution_numerator;
	float error_damage_solution_denominator;


	Vector<double> solution_damage_difference;
	Vector<double> solution_elastic_difference;
	Vector<double> H_vector_new;
	Vector<double> H_vector;

	//objects for load displacement curve
	Vector<double> force_vector;
	Vector<double> load_values;
	Vector<double> displacement_values;
	SparseMatrix<double> copy_of_system_matrix_elastic;

};
//for E=210GPa and nu=0.3, lambda= 121.1538 and mu=80.7692
double lambda( const Point<2> & p,
		const unsigned int component = 0 );
double lambda(const Point<2> &p,
		const unsigned int )
{
	double return_value = 0.0;
	for (unsigned int i = 0; i < 2; ++i)

		return_value = 121.1538 + p[0]-p[0];
	return return_value;
}
double mu( const Point<2> & p,
		const unsigned int component = 0 );
double mu(const Point<2> &p,
		const unsigned int )
{
	double return_value = 0.0;
	for (unsigned int i = 0; i < 2; ++i)

		return_value = 80.7692 + p[0]-p[0];
	return return_value;
}

//  declarations
double Conductivity_damage( const Point<2> & p,
		const unsigned int component = 0 );//const override;
class BoundaryValuesDamage : public Function<2>
{
public:
	virtual double value(const Point<2> & p,
			const unsigned int component = 0) const override;
};
class FluxDamage : public Function<2>
{
public:
	virtual double value(const Point<2> & p,
			const unsigned int component = 0) const override;
};

// Function definitions

double Conductivity_damage(const Point<2> &p,
		const unsigned int ) //const
{
	double return_value = 0.0;
	for (unsigned int i = 0; i < 2; ++i)
		return_value = p[0]-p[0]+1;
	return return_value;
}
double FluxDamage::value(const Point<2> &p,
		const unsigned int ) const
{
	return p[0]- p[0] + 0;
}
double BoundaryValuesDamage::value(const Point<2> &p,
		const unsigned int) const
{
	return p[0]+p[1];
}

void right_hand_side_elastic(const std::vector<Point<2>> &points,
		std::vector<Tensor<1, 2>> &  values)
{
	AssertDimension(values.size(), points.size());
	/*std::cout <<"size of point for rhs" << points.size() <<std::endl;
    std::cout <<"size of values for rhs" << values.size() <<std::endl;*/
	Assert(2 >= 2, ExcNotImplemented());


	for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
	{
		values[point_n][0] = 0; //x component of body force
		values[point_n][1] = 0; //y component of body force
	}

}
void Traction_elastic(const std::vector<Point<2>> &points,
		std::vector<Tensor<1, 2>> &  values)

{
	//AssertDimension(values.size(), points.size());
	//Assert(2 >= 2, ExcNotImplemented());


	for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
	{
		values[point_n][0] = 0; //x component of traction
		values[point_n][1] = 0; //y component of traction
	}

}


PhaseField::PhaseField()
: fe_damage(1), dof_handler_damage(triangulation)
,fe_elastic(FE_Q<2>(1),2),dof_handler_elastic(triangulation)


{}




float PhaseField::H_plus(Vector<double> & solution_elastic
		, const auto cell,const unsigned int q_point,const FEValues<2> & fe_values_damage)
{
	/*QGauss<2> quadrature_formula_damage(fe_damage.degree + 1);

	FEValues<2> fe_values_damage(fe_damage,
			quadrature_formula_damage,
			update_gradients |
			update_quadrature_points );*/

	/*fe_values_damage.reinit(cell);*/

	int node = 0;
	float e_xx = 0.000;
	float e_yy = 0.000;
	float e_xy = 0.000;
	//float e_xx_plus = 0.000;
	//float e_yy_plus = 0.000;
	//float e_xy_plus = 0.000;
	for (const auto vertex : cell->vertex_indices())
	{
		int a = (cell->vertex_dof_index(vertex, 0));
		e_xx = e_xx + solution_elastic[a*2]*fe_values_damage.shape_grad(node, q_point)[0];
		e_yy = e_yy + solution_elastic[a*2+1]*fe_values_damage.shape_grad(node, q_point)[1];
		e_xy = e_xy +0.5*(solution_elastic[a*2]*fe_values_damage.shape_grad(node, q_point)[1]
																						   +solution_elastic[a*2+1]*fe_values_damage.shape_grad(node, q_point)[0]);
		node = node +1;
	}
	/*if (e_xx > 0)
	{
		e_xx_plus = e_xx;
	}
	if (e_xy > 0)
	{
		e_xy_plus = e_xy;
	}
	if (e_yy > 0)
	{
		e_yy_plus = e_yy;
	}
*/
	float tr_strain, det_strain, vap1,vap2,Apos,Bpos;/*Aneg,Bneg*/
	tr_strain = e_xx + e_yy;
	det_strain = (e_xx*e_yy) - (e_xy*e_xy);
	vap1 = tr_strain/2 + pow(((pow(tr_strain,2))/4 - det_strain),0.5);
	vap2 = tr_strain/2 - pow(((pow(tr_strain,2))/4 - det_strain),0.5);

	Apos = pow((0.5*(tr_strain + std::fabs(tr_strain))),2);
	//Aneg = pow((0.5*(tr_strain - std::fabs(tr_strain))),2);
	Bpos = pow((0.5*(vap1 + std::fabs(vap1))),2) + pow((0.5*(vap2 + std::fabs(vap2))),2);
	//Bneg = pow((0.5*(vap1 - std::fabs(vap1))),2) + pow((0.5*(vap2 - std::fabs(vap2))),2);

	const auto &x_q = fe_values_damage.quadrature_point(q_point);
	float H_plus_val;
	H_plus_val = 0.5*lambda(x_q)*Apos +  mu(x_q)*Bpos;
	return H_plus_val;
}

float PhaseField::damage_gauss_pt(Vector<double> & solution_damage,const auto cell
		,const unsigned int q_point
		,const FEValues<2> & fe_values_elastic
)


{
	/*QGauss<2> quadrature_formula_elastic(fe_elastic.degree + 1);

	FEValues<2> fe_values_elastic(fe_elastic,
			quadrature_formula_elastic,
			update_values);*/

	/*fe_values_elastic.reinit(cell);*/
	int node = 0;
	float d = 0;

	for (const auto vertex : cell->vertex_indices())
	{
		int a = (int) ((cell->vertex_dof_index(vertex, 0))/2);

		/*const unsigned int component_i =
				fe_elastic.system_to_component_index(2*node).first;*/

		d = d + solution_damage[a]*fe_values_elastic.shape_value(2*node, q_point);

		node = node +1;
	}

	return d;

}

void PhaseField::setup_system_elastic()
{
	Timer timer_setup_system_elastic; // creating a timer also starts it

	solution_elastic.reinit(dof_handler_elastic.n_dofs());

	system_rhs_elastic.reinit(dof_handler_elastic.n_dofs());


	DynamicSparsityPattern dsp(dof_handler_elastic.n_dofs(), dof_handler_elastic.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler_elastic,
			dsp
			//                                    constraints,
	/*keep_constrained_dofs = */ /*false*/);
	sparsity_pattern_elastic.copy_from(dsp);

	system_matrix_elastic.reinit(sparsity_pattern_elastic);

	/*const IndexSet dofs_applied_disp = DoFTools::extract_boundary_dofs(dof_handler_elastic,	component_mask = u_x_mask,4);

	const types::global_dof_index first_boundary_dof =
			dofs_applied_disp.nth_index_in_set(0);
	for (int i = 0; i < dof_handler_elastic.n_dofs(); i++)
	{
		std::cout << dofs_applied_disp.nth_index_in_set(i) << std::endl;
	}
*/
	timer_setup_system_elastic.stop();

	std::cout << "Elapsed wall time  setup_system_elastic: " << timer_setup_system_elastic.wall_time() << " seconds.\n";

	// reset timer for the next thing it shall do
	timer_setup_system_elastic.reset();


}
void PhaseField::setup_boundary_values_elastic()
{

	Timer timer_setup_boundary_values_elastic; // creating a timer also starts it

	boundary_values_elastic.clear();
	for (const auto &cell : dof_handler_elastic.active_cell_iterators())
	{
		for (const auto &face : cell->face_iterators())
		{
			if (face->at_boundary())
			{
				const auto center = face->center();
				if (std::fabs(center(1) - (0)) < 1e-12) //face lies at bottom edge
				{


					for (const auto vertex_number : cell->vertex_indices())

					{
						const auto vert = cell->vertex(vertex_number);
						if (std::fabs(vert(1) - (0)) < 1e-12)
						{   types::global_dof_index x_displacement =
								cell->vertex_dof_index(vertex_number, 0);
						types::global_dof_index y_displacement =
								cell->vertex_dof_index(vertex_number, 1);
						boundary_values_elastic[x_displacement] = 0; //fixed in x
						boundary_values_elastic[y_displacement] = 0; //fixed in y
						}




					}




				}
			}
		}
	}
	timer_setup_boundary_values_elastic.stop();

	std::cout << "Elapsed wall time setup_boundary_values_elastic: " << timer_setup_boundary_values_elastic.wall_time() << " seconds.\n";

	// reset timer for the next thing it shall do
	timer_setup_boundary_values_elastic.reset();
}


void PhaseField::assemble_system_elastic(int t)

{
	Timer timer_assemble_system_elastic; // creating a timer also starts it

	QGauss<2> quadrature_formula_elastic(fe_elastic.degree + 1);

	/* face_quadrature_formula_elastic is needed only if traction BC is specified for elasticity equation*/

	/*QGauss<1> face_quadrature_formula_elastic(fe_elastic.degree + 1);
	 */

	/*FEValues<dim>     fe_values(mapping,
                           fe,
                           quadrature_formula,
                           update_values | update_gradients |
                             update_quadrature_points | update_JxW_values);*/



	/*FEFaceValues are needed only if traction BC is specified for elasticity equation*/
	/*FEFaceValues<2> fe_face_values_elastic(//mapping,
    			  fe_elastic,
				  face_quadrature_formula_elastic,
				  update_values | update_quadrature_points |
				  update_normal_vectors |
				  update_JxW_values);
	 */


	FEValues<2> fe_values_elastic(fe_elastic,
			quadrature_formula_elastic,
			update_values | update_gradients |
			update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe_elastic.n_dofs_per_cell();
	const unsigned int n_q_points    = quadrature_formula_elastic.size();

	FullMatrix<double> cell_matrix_elastic(dofs_per_cell, dofs_per_cell);
	Vector<double>     cell_rhs_elastic(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	std::vector<Tensor<1, 2>> rhs_values_elastic(n_q_points);
	std::vector<Tensor<1, 2>> traction_values_elastic(n_q_points);


	for (const auto &cell : dof_handler_elastic.active_cell_iterators())
	{

		cell_matrix_elastic = 0;
		cell_rhs_elastic    = 0;
		fe_values_elastic.reinit(cell);
		// std::cout<<cell->get_fe().get_name()<<std::endl;

		right_hand_side_elastic(fe_values_elastic.get_quadrature_points(), rhs_values_elastic);
		Traction_elastic(fe_values_elastic.get_quadrature_points(),traction_values_elastic );
		for (const unsigned int q_point :
				fe_values_elastic.quadrature_point_indices())
		{
			float d = damage_gauss_pt(solution_damage,cell,q_point, fe_values_elastic
			);
			for (const unsigned int i : fe_values_elastic.dof_indices())

			{const unsigned int component_i =
					fe_elastic.system_to_component_index(i).first;

			for (const unsigned int j : fe_values_elastic.dof_indices())
			{
				const unsigned int component_j =
						fe_elastic.system_to_component_index(j).first;

				{ const auto &x_q = fe_values_elastic.quadrature_point(q_point);

				cell_matrix_elastic(i, j) +=
						pow((1-d),2)*(                                                  //
								(fe_values_elastic.shape_grad(i, q_point)[component_i] * //
										fe_values_elastic.shape_grad(j, q_point)[component_j] * //
										lambda(x_q))                         //
										+                                                //
										(fe_values_elastic.shape_grad(i, q_point)[component_j] * //
												fe_values_elastic.shape_grad(j, q_point)[component_i] * //
												mu(x_q))                             //
												+                                                //
												((component_i == component_j) ?        //
														(fe_values_elastic.shape_grad(i, q_point) * //
																fe_values_elastic.shape_grad(j, q_point) * //
																mu(x_q)) :              //
																0)                                  //
						) *                                    //
						fe_values_elastic.JxW(q_point);                  //
				}
			}
			}
		}


		for (const unsigned int i : fe_values_elastic.dof_indices())
		{
			const unsigned int component_i =
					fe_elastic.system_to_component_index(i).first;

			for (const unsigned int q_point :
					fe_values_elastic.quadrature_point_indices())
				cell_rhs_elastic(i) += fe_values_elastic.shape_value(i, q_point) *
				rhs_values_elastic[q_point][component_i] *
				fe_values_elastic.JxW(q_point);
		}

		// traction contribution to rhs
		/* for (const auto &face : cell->face_iterators())
    		   {
    			   if (face->at_boundary() &&
    					   face->boundary_id() == 10)

    			   {
    				   fe_face_values_elastic.reinit(cell, face);
    				   for (const auto i : fe_face_values_elastic.dof_indices())

    				   {const unsigned int component_i =
    						   fe_elastic.system_to_component_index(i).first;
    				   for (const auto face_q_point :
    						   fe_face_values_elastic.quadrature_point_indices())
    				   {

    					   cell_rhs_elastic(i) +=
    							   fe_face_values_elastic.shape_value( i, face_q_point)
		 *(traction_values_elastic[face_q_point][component_i])
		 *fe_face_values_elastic.JxW(face_q_point);



    				   }
    				   }
    			   }
    		   }*/


		/*Adding the local k and local f to global k and global f*/
		// This part of code is written within the element loop
		cell->get_dof_indices(local_dof_indices);
		for (const unsigned int i : fe_values_elastic.dof_indices())
		{
			for (const unsigned int j : fe_values_elastic.dof_indices())
				system_matrix_elastic.add(local_dof_indices[i],
						local_dof_indices[j],
						cell_matrix_elastic(i, j));

			system_rhs_elastic(local_dof_indices[i]) += cell_rhs_elastic(i);
		}



	}
	  copy_of_system_matrix_elastic.copy_from (system_matrix_elastic);

	/*Applying bc after assembling*/

	FEValuesExtractors::Scalar u_x(0);
	FEValuesExtractors::Scalar u_y(1);

	ComponentMask u_x_mask = fe_elastic.component_mask(u_x);
	ComponentMask u_y_mask = fe_elastic.component_mask(u_y);

	// Shear Bcs on top face
	double u_x_values = 0.0001*t;
	double u_y_values = 0.000;
	//Tension Bcs on top face
	/*double u_y_values =0.0001*t;*/


	//Tension test BC:
	/* VectorTools::interpolate_boundary_values(dof_handler_elastic,
    			  4,
				  Functions::ConstantFunction<2>(u_y_values,2),
				  boundary_values_elastic,u_y_mask
    	  );*/

	//Shear test BC:

	VectorTools::interpolate_boundary_values(dof_handler_elastic,
			4,
			Functions::ConstantFunction<2>(u_y_values,2),
			boundary_values_elastic,u_y_mask
	);

	VectorTools::interpolate_boundary_values(dof_handler_elastic,
			4,
			Functions::ConstantFunction<2>(u_x_values,2),
			boundary_values_elastic,u_x_mask
	);
	/*for (int i =0; i < dof_handler_elastic.n_dofs() ; i++)

		{
		for (int j =0; j < dof_handler_elastic.n_dofs() ; j++)

		{
	std::cout << " before applybc K" << "" << i << "" << j <<  std::endl;

	 std::cout << system_matrix_elastic.operator()(12,4) << std::endl;

		}
		}*/
	/*std::cout << " before applybc K_12_4" <<  std::endl;

		 std::cout << system_matrix_elastic.operator()(12,4) << std::endl;*/
	std::cout << " before applybc K_4_12" <<  std::endl;

			 std::cout << system_matrix_elastic.operator()(4,12) << std::endl;
	MatrixTools::apply_boundary_values(boundary_values_elastic,
			system_matrix_elastic,
			solution_elastic,
			system_rhs_elastic, false);
	std::cout << "system_matrix_elastic(4,12)after applybc" << std::endl;

	std::cout << system_matrix_elastic.operator()(4,12) << std::endl;

	std::cout << "system_matrix_elastic(12, 13) after apply bc" << std::endl;

		std::cout << system_matrix_elastic.operator()(12, 13) << std::endl;

	timer_assemble_system_elastic.stop();

	std::cout << "Elapsed wall time assemble_system_elastic: " << timer_assemble_system_elastic.wall_time() << " seconds.\n";

	// reset timer for the next thing it shall do
	timer_assemble_system_elastic.reset();


}

void PhaseField::solve_elastic()
{
	Timer timer_solve_elastic; // creating a timer also starts it
	std::cout << "system_matrix_elastic(4,12) before solve call" << std::endl;

		std::cout << system_matrix_elastic.operator()(4,12) << std::endl;


	std::cout << "Solving linear system for elasticity... ";




	// Iterative solver
	SolverControl            solver_control(1000, 1e-6* system_rhs_elastic.l2_norm());
	SolverCG<Vector<double>> cg(solver_control);

	PreconditionSSOR<SparseMatrix<double>> preconditioner;
	preconditioner.initialize(system_matrix_elastic, 1.2);

	cg.solve(system_matrix_elastic, solution_elastic, system_rhs_elastic, preconditioner);
	// Direct solver



	/* SparseDirectUMFPACK A_direct;
   A_direct.initialize(system_matrix_elastic);

   A_direct.vmult(solution_elastic, system_rhs_elastic);



	 */

	/*for (int i=0; i < dof_handler_elastic.n_dofs(); i++)
  	   std::cout << solution_elastic[i] << std::endl;
	 */

	std::cout << "system_matrix_elastic(4,12)after solve" << std::endl;

		std::cout << system_matrix_elastic.operator()(4,12) << std::endl;


	timer_solve_elastic.stop();
	std::cout << "Elapsed wall time _solve_elastic: " << timer_solve_elastic.wall_time() << " seconds.\n";

	// reset timer for the next thing it shall do
	timer_solve_elastic.reset();
}

void PhaseField::setup_boundary_values_damage()
{

	Timer timer_setup_boundary_values_damage; // creating a timer also starts it

	boundary_values_damage.clear();
	for (const auto &cell : dof_handler_damage.active_cell_iterators())
	{
		/*for (const auto &face : cell->face_iterators())
		{
			for (const auto vertex_number : cell->vertex_indices())
			{
				const auto vert = cell->vertex(vertex_number);
				if ((vert(0) - 0 < 1e-12) &&
						(std::fabs(vert(1) - 0) < 1e-12)) // nodes on initial damage line
				{   types::global_dof_index damage =
						cell->vertex_dof_index(vertex_number, 0);

				boundary_values_damage[damage] = 1; //prescribed damage
				}
			}
		}*/
		for (const auto vertex_number : cell->vertex_indices())
					{
						const auto vert = cell->vertex(vertex_number);
						if ((vert(0) - 0.5 < 1e-12) &&
								(std::fabs(vert(1) - (0.5)) < 1e-12)) // nodes on initial damage line
						{   types::global_dof_index damage =
								cell->vertex_dof_index(vertex_number, 0);

						boundary_values_damage[damage] = 1; //prescribed damage
						}
					}
	}

	timer_setup_boundary_values_damage.stop();
	std::cout << "Elapsed wall time _setup_boundary_values_damage: " << timer_setup_boundary_values_damage.wall_time() << " seconds.\n";

	// reset timer for the next thing it shall do
	timer_setup_boundary_values_damage.reset();
}

void PhaseField::setup_system_damage()
{
	Timer timer_setup_system_damage; // creating a timer also starts it

	DynamicSparsityPattern dsp(dof_handler_damage.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler_damage, dsp);
	sparsity_pattern_damage.copy_from(dsp);

	system_matrix_damage.reinit(sparsity_pattern_damage);

	solution_damage.reinit(dof_handler_damage.n_dofs());
	system_rhs_damage.reinit(dof_handler_damage.n_dofs());

	timer_setup_system_damage.stop();
	std::cout << "Elapsed wall time:_setup_system_damage " << timer_setup_system_damage.wall_time() << " seconds.\n";

	// reset timer for the next thing it shall do
	timer_setup_system_damage.reset();
}



void PhaseField::assemble_system_damage()
{

	Timer timer_assemble_system_damage; // creating a timer also starts it

	QGauss<2> quadrature_formula_damage(fe_damage.degree + 1);
	/*QGauss<2 - 1> face_quadrature_formula_damage(fe_damage.degree + 1);
	 */ FluxDamage flux_damage;
	 FEValues<2> fe_values_damage(fe_damage,
			 quadrature_formula_damage,
			 update_values | update_gradients | update_JxW_values | update_quadrature_points);

	 /*FEFaceValues<2> fe_face_values_damage(fe_damage,
			   face_quadrature_formula_damage,
			   update_values | update_quadrature_points |
			   update_normal_vectors |
			   update_JxW_values);
	  */
	 const unsigned int dofs_per_cell = fe_damage.n_dofs_per_cell();

	 FullMatrix<double> cell_matrix_damage(dofs_per_cell, dofs_per_cell);
	 Vector<double>     cell_rhs_damage(dofs_per_cell);

	 std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	 int cell_number =0;

	 for (const auto &cell : dof_handler_damage.active_cell_iterators())
	 {

		 fe_values_damage.reinit(cell);
		 cell_matrix_damage = 0;
		 cell_rhs_damage    = 0;
		 //std::cout<<cell->get_fe().get_name()<<std::endl;

		 float gc = 0.0027;//energy release rate
		 float l = 0.015;
		 float H;

		 for (const unsigned int q_index : fe_values_damage.quadrature_point_indices())
		 {	float H_call = H_plus(solution_elastic,cell,q_index, fe_values_damage);


		 if (H_call > H_vector[4*cell_number + q_index])
		 {
			 H = H_call;
		 }
		 else
		 {
			 H = H_vector[4*cell_number + q_index];
		 }
		 H_vector_new[4*cell_number + q_index] = H;
		 for (const unsigned int i : fe_values_damage.dof_indices())
		 {


			 for (const unsigned int j : fe_values_damage.dof_indices())
			 {

				 const auto &x_q = fe_values_damage.quadrature_point(q_index);


				 cell_matrix_damage(i, j) +=
						 // contribution to stiffness from -laplace u term
						 Conductivity_damage(x_q)*fe_values_damage.shape_grad(i, q_index) * // kappa*grad phi_i(x_q)
						 fe_values_damage.shape_grad(j, q_index) * // grad phi_j(x_q)
						 fe_values_damage.JxW(q_index)    // dx
						 +
						 // Contribution to stiffness from u term

						 ((1+(2*l*H)/gc)*(1/pow(l,2))*fe_values_damage.shape_value(i, q_index) *    // phi_i(x_q)
								 fe_values_damage.shape_value(j, q_index) * // phi_j(x_q)
								 fe_values_damage.JxW(q_index));             // dx
			 }
			 const auto &x_q = fe_values_damage.quadrature_point(q_index);
			 cell_rhs_damage(i) += (fe_values_damage.shape_value(i, q_index) * // phi_i(x_q)
					 (2/(l*gc))* H*
					 fe_values_damage.JxW(q_index));            // dx
		 }
		 }
		 /* ADDED FOR NEUMAN CONTRIBUTION*/
		 /* for (const auto &face : cell->face_iterators())
		   {
			   if (face->at_boundary() && (face->boundary_id() == (1 || 2 || 3 || 4)))
			   {
				   fe_face_values_damage.reinit(cell, face);
				   for (const unsigned int q_indexs : fe_face_values_damage.quadrature_point_indices())
				   {

					   const auto &q_points = fe_face_values_damage.quadrature_point(q_indexs);
					   const double neumann_value = flux_damage.value(q_points);

					   for (unsigned int i = 0; i < dofs_per_cell; ++i)


						   cell_rhs_damage(i) +=
								   (fe_face_values_damage.shape_value(i, q_indexs) * // phi_i(x_q)
										   neumann_value *                          // g(x_q)
										   fe_face_values_damage.JxW(q_indexs));   // dx

				   }
			   }
		   }*/
		 /*end of neumann contribution */

		 /*Adding the local k and local f to global k and global f*/
		 // This part of code is written within the element loop

		 cell->get_dof_indices(local_dof_indices);
		 for (const unsigned int i : fe_values_damage.dof_indices())
		 {
			 for (const unsigned int j : fe_values_damage.dof_indices())
				 system_matrix_damage.add(local_dof_indices[i],
						 local_dof_indices[j],
						 cell_matrix_damage(i, j));

			 system_rhs_damage(local_dof_indices[i]) += cell_rhs_damage(i);
		 }
		 cell_number = cell_number + 1;




	 }


	VectorTools::interpolate_boundary_values(dof_handler_damage,
			 6,// 6 is the boundary id
			 BoundaryValuesDamage(),
			 boundary_values_damage);
	 // To set a different value of Dirichlet BC
	 VectorTools::interpolate_boundary_values(dof_handler_damage,
			 8,
			 BoundaryValuesDamage(),//Functions::ConstantFunction<2>(1),
			 boundary_values_damage);


	 MatrixTools::apply_boundary_values(boundary_values_damage,
			 system_matrix_damage,
			 solution_damage,
			 system_rhs_damage);

	 timer_assemble_system_damage.stop();

	 std::cout << "Elapsed wall time: _assemble_system_damage" << timer_assemble_system_damage.wall_time() << " seconds.\n";

	 // reset timer for the next thing it shall do
	 timer_assemble_system_damage.reset();

}
void PhaseField::solve_damage()
{
	Timer timer_solve_damage;
	std::cout << "Solving linear system for damage... ";

	// Iterative solver
	SolverControl            solver_control_damage(1000, 1e-6* system_rhs_damage.l2_norm());
	SolverCG<Vector<double>> cg_damage(solver_control_damage);

	PreconditionSSOR<SparseMatrix<double>> preconditioner;
	preconditioner.initialize(system_matrix_damage, 1.2);

	cg_damage.solve(system_matrix_damage, solution_damage, system_rhs_damage, preconditioner);
	// std::cout << "You are in PhaseField::solve_damage" << std::endl;


	/*

	   SparseDirectUMFPACK A_direct;
	   A_direct.initialize(system_matrix_damage);

	   A_direct.vmult(solution_damage, system_rhs_damage);
	 */


	timer_solve_damage.stop();
	std::cout << "Elapsed wall time: _solve_damage" << timer_solve_damage.wall_time() << " seconds.\n";

	// reset timer for the next thing it shall do
	timer_solve_damage.reset();
}




void PhaseField::elastic_solver(int t)
{
	setup_boundary_values_elastic();
	setup_system_elastic();
	assemble_system_elastic(t);
	solve_elastic();

}

void PhaseField::output_results(int t) const
{
	std::vector<std::string> displacement_names(2, "displacement");
	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	displacement_component_interpretation(
			2, DataComponentInterpretation::component_is_part_of_vector);

	DataOut<2> data_out_phasefield;
	data_out_phasefield.add_data_vector(dof_handler_elastic,
			solution_elastic,
			displacement_names,
			displacement_component_interpretation);
	data_out_phasefield.add_data_vector(dof_handler_damage,
			solution_damage,
			"damage");
	data_out_phasefield.build_patches(std::min(1, 1));//1,1 are degrees of two fnite elements

	std::ofstream output("solution-" +
			Utilities::int_to_string(t, 4) + ".vtk");
	data_out_phasefield.write_vtk(output);

}



void PhaseField::damage_solver()
{
	setup_boundary_values_damage();
	setup_system_damage();
	assemble_system_damage();
	solve_damage();
}
void PhaseField::run()
{ std::cout << "You are in PhaseField::run" << std::endl;
//GridGenerator::hyper_cube(triangulation, -1, 1);
Point<2> point_1, point_2;
point_1(0) = 0;
point_1(1) = 0;
point_2(0) = 1;
point_2(1)= 1;
GridGenerator::hyper_rectangle(triangulation, point_1, point_2);


for (const auto &cell : triangulation.active_cell_iterators())
{
	for (const auto &face : cell->face_iterators())
	{
		if (face->at_boundary())
		{
			const auto center = face->center();
			if (std::fabs(center(0) - (0)) < 1e-12) //left face
			{

				face->set_boundary_id(1);

			}
			else if (std::fabs(center(0) - 1) < 1e-12) //right face
			{

				face->set_boundary_id(2);

			}
			else if (std::fabs(center(1) - (0)) < 1e-12) //bottom face
			{

				face->set_boundary_id(3);

			}
			else if (std::fabs(center(1) - 1) < 1e-12) //top face
			{

				face->set_boundary_id(4);

			}

		}
	}
}
triangulation.refine_global(1);

std::cout << "   Number of active cells:       "
		<< triangulation.n_active_cells() << std::endl;

dof_handler_damage.distribute_dofs(fe_damage);
dof_handler_elastic.distribute_dofs(fe_elastic);
std::cout << "   Number of degrees of freedom for damage mesh: " << dof_handler_damage.n_dofs()
	                				   << std::endl;
std::cout << "   Number of degrees of freedom for elastic mesh: " << dof_handler_elastic.n_dofs()
	   	                				   << std::endl;

/*Vector<double> solution_damage(dof_handler_damage.n_dofs());
	   (This way solution_damage cannot be initialized.
	Instead it is as if we are defining a new local variable solution_damage which is not accessible to other functions)
 */
solution_damage = Vector<double>(dof_handler_damage.n_dofs());

for (const auto &cell : dof_handler_damage.active_cell_iterators())
{
	for (const auto vertex_number : cell->vertex_indices())
	{
		const auto vert = cell->vertex(vertex_number);
		int a =  cell->vertex_dof_index(vertex_number, 0);//a gives the global dof corresponding to first dof of current cell,vertex_number
		if ( ((vert(0) - 0) < 1e-12) && (std::fabs(vert(1) - 0) < 1e-12))
		{
			solution_damage[a] = 1;
		}
		else
		{
			solution_damage[a] = 0;
		}
	}
}
solution_damage_old =   Vector<double>(dof_handler_damage.n_dofs());
solution_elastic_old =  Vector<double>(2*dof_handler_damage.n_dofs());

solution_damage_difference = Vector<double>(dof_handler_damage.n_dofs());
solution_elastic_difference = Vector<double>(2*dof_handler_damage.n_dofs());

H_vector = Vector<double> (4*(triangulation.n_active_cells()));
H_vector_new = Vector<double> (4*triangulation.n_active_cells());
int t_max = 2;
load_values.reinit(t_max);
displacement_values.reinit(t_max);
FEValuesExtractors::Scalar u_x(0);
FEValuesExtractors::Scalar u_y(1);

ComponentMask u_x_mask = fe_elastic.component_mask(u_x);
ComponentMask u_y_mask = fe_elastic.component_mask(u_y);


/*std::cout << "hi" << std::endl;*/

/*const IndexSet dofs_applied_disp = DoFTools::extract_boundary_dofs(dof_handler_elastic,	component_mask = u_x_mask,4);*/

/*for (types::global_dof_index i : dofs_applied_disp) // i takes the values of global dofs on the boundary
	{
		std::cout << i << std::endl;
	}*/

	/*std::cout << "bye" << std::endl;*/
for (int t=1;t<t_max;t++)
{
	std::cout << " \n \n load increment number : " << t << std::endl;
	int iteration = 0;
	int stoppingCriterion = 0;
	while(stoppingCriterion == 0 && iteration <100)

	{
		iteration = iteration + 1;
		std::cout << " \n iteration number:" << iteration << std::endl;
		elastic_solver(t);
		std::cout << "Out of elastic solver" << std::endl;
		damage_solver();
		std::cout << "Out of damage solver" << std::endl;

		if (iteration == 1)
		{
			solution_damage_old  = solution_damage;
			solution_elastic_old =  solution_elastic;

		}
		else
		{
			for(unsigned int i = 0; i < dof_handler_damage.n_dofs(); i++ )
			{
				solution_damage_difference[i] =  solution_damage_old[i] - solution_damage[i];
				solution_elastic_difference[2*i] =  solution_elastic_old[2*i] - solution_elastic[2*i];
				solution_elastic_difference[(2*i) +1] =  solution_elastic_old[(2*i)+1] - solution_elastic[(2*i)+1];

			}

			error_damage_solution_numerator = 0;
			error_elastic_solution_numerator = 0;
			for (unsigned int i = 0; i < dof_handler_damage.n_dofs(); i++  )
			{

				error_damage_solution_numerator = error_damage_solution_numerator +(solution_damage_difference[i]*solution_damage_difference[i]);

				error_elastic_solution_numerator = error_elastic_solution_numerator +(solution_elastic_difference[2*i]*solution_elastic_difference[2*i])
				   											   + (solution_elastic_difference[(2*i)+1]*solution_elastic_difference[(2*i)+1]);
			}
			error_damage_solution_numerator = pow(error_damage_solution_numerator,0.5);
			error_elastic_solution_numerator = pow(error_elastic_solution_numerator,0.5);
			error_damage_solution_denominator = 0;
			error_elastic_solution_denominator = 0;
			for (unsigned int i = 0; i < dof_handler_damage.n_dofs(); i++  )
			{

				error_damage_solution_denominator = error_damage_solution_denominator +(solution_damage[i]*solution_damage[i]);

				error_elastic_solution_denominator = error_elastic_solution_denominator +(solution_elastic[2*i]*solution_elastic[2*i])
				   											   + (solution_elastic[(2*i)+1]*solution_elastic[(2*i)+1]);
			}
			error_damage_solution_denominator = pow(error_damage_solution_denominator,0.5);
			error_elastic_solution_denominator = pow(error_elastic_solution_denominator,0.5);


			error_damage_solution = error_damage_solution_numerator/error_damage_solution_denominator;
			error_elastic_solution = error_elastic_solution_numerator/error_elastic_solution_denominator;

			if ((error_elastic_solution < 0.01) && (error_damage_solution < 0.01))
			{
				stoppingCriterion = 1;
			}
			else
			{
				stoppingCriterion = 0;
			}
			solution_damage_old  = solution_damage;
			solution_elastic_old =  solution_elastic;


		}



	}

	for (unsigned int i=0;i< 4*triangulation.n_active_cells();i++)
	{
		H_vector[i] = H_vector_new[i];
	}

	{
		if (t==1 || t==40 || t==80 || t==100 || t==140 || t==160 || t==180 || t==200)
		{
		output_results(t);
		}

	}




	//load displacement curves:
	const IndexSet dofs_applied_disp = DoFTools::extract_boundary_dofs(dof_handler_elastic,	/*component_mask =*/ u_x_mask ,/*boundary_ids =*/ {3});


	for (types::global_dof_index i : dofs_applied_disp) // i takes the values of global dofs on the boundary

		{
			std::cout << solution_elastic[i] << std::endl;
		}


	float load = 0;

	force_vector.reinit(dof_handler_elastic.n_dofs());

	copy_of_system_matrix_elastic.vmult(force_vector,solution_elastic);
	for (types::global_dof_index i : dofs_applied_disp)

	{

			std::cout <<  i << "-th component of solution_vector" << ""  << std::endl;
			std::cout <<solution_elastic[i] << std::endl;

	}
	/*std::cout << "system_matrix_elastic components" << std::endl;
	std::cout << system_matrix_elastic.operator()(12,4) << std::endl;
	std::cout << system_matrix_elastic.operator()(14,4) << std::endl;
	std::cout << system_matrix_elastic.operator()(12,6) << std::endl;
	std::cout << system_matrix_elastic.operator()(14,6) << std::endl;
	std::cout << system_matrix_elastic.operator()(16,6) << std::endl;
	std::cout << system_matrix_elastic.operator()(14,10) << std::endl;
	std::cout << system_matrix_elastic.operator()(16,10) << std::endl;
	std::cout << system_matrix_elastic.operator()(12,12) << std::endl;
	std::cout << system_matrix_elastic.operator()(14,12) << std::endl;

	std::cout << system_matrix_elastic.operator()(12,14) << std::endl;
	std::cout << system_matrix_elastic.operator()(14,14) << std::endl;
	std::cout << system_matrix_elastic.operator()(16,14) << std::endl;

	std::cout << system_matrix_elastic.operator()(14,16) << std::endl;
	std::cout << system_matrix_elastic.operator()(16,16) << std::endl;

	std::cout << system_matrix_elastic.operator()(14,5) << std::endl;*/
	std::cout << system_matrix_elastic.operator()(4,12) << std::endl;


	for (types::global_dof_index i : dofs_applied_disp)
	{
		std::cout << i << std::endl;
		std::cout <<  i << "-th component of force_vector" << ""  << std::endl;
		std::cout << force_vector[i] << std::endl;
	load = load + force_vector[i];
	}
	load_values[t] = load;
	displacement_values[t] = 0.0001*t;
	std::cout << "load= " << load_values[t] << std::endl;
	std::cout << "displacement= " << displacement_values[t] << std::endl;
	float f_twelve = 0;
	std::vector<int> v = { 4,6,12,14,5,7,13,15 };
	for (auto i:v)
	{
     f_twelve = f_twelve + (system_matrix_elastic.operator()(12,i))*solution_elastic[i] ;
	}
	std::cout << f_twelve << std::endl;
	/*float f_eight =0;
		for (int i=0; i <18; i++)
		{
	     f_eight = f_eight + system_matrix_elastic.operator()(14,i)*solution_elastic[i] ;
		}
		float f_nine =0;
			for (int i=0; i <18; i++)
			{
		     f_nine = f_nine + system_matrix_elastic.operator()(16,i)*solution_elastic[i] ;
			}
 float LOAD = f_seven + f_eight + f_nine;
 std::cout  << "LOAD = " << std::endl;
	std::cout  << LOAD << std::endl;*/




}


}
}
int main()

{
	step203::PhaseField phasefield;

	phasefield.run();

	return 0;
}
