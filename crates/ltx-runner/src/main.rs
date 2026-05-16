use ltx_core::reset_plan::restart_plan;

fn main() {
    let plan = restart_plan();
    println!("{}", plan.primary_goal);
}
