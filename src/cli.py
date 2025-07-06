import click
import SimulateDrawmaton as sim
import utilities as util
import numpy as np
import os

def get_absolute_path(path):
    """Convert relative path to absolute path"""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.getcwd(), path))

# Common options for dimensions
def dimension_options(f):
    f = click.option('--l1', '-l1', type=float, default=5.8, help='L1 dimension')(f)
    f = click.option('--l2', '-l2', type=float, default=16.6, help='L2 dimension')(f)
    f = click.option('--l3', '-l3', type=float, default=24.3, help='L3 dimension')(f)
    f = click.option('--gx', '-gx', type=float, default=-4.5, help='Gx dimension')(f)
    f = click.option('--gy', '-gy', type=float, default=16.0, help='Gy dimension')(f)
    return f

@click.group()
def cli():
    """SEM-Drawmaton: Da Vinci-Inspired Mechanical Drawing Machine Simulator

    Create, simulate, and animate mechanical drawings using a system of linkages and rotors,
    inspired by Leonardo da Vinci's original Drawmaton design."""
    pass

@cli.group()
def create():
    """Create simulation files from different input sources"""
    pass

@create.command()
@dimension_options
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input image file')
@click.option('--output', '-o', type=click.Path(), required=True, help='Output simulation file')
@click.option('--target-x', type=float, default=13.4, help='Target X position')
@click.option('--target-y', type=float, default=27.1, help='Target Y position')
@click.option('--target-w', type=float, default=16.3, help='Target width')
@click.option('--target-h', type=float, default=16.3, help='Target height')
@click.option('--interactive/--non-interactive', default=False, help='Enable/disable interactive contour selection')
def image(l1, l2, l3, gx, gy, input, output, target_x, target_y, target_w, target_h, interactive):
    """Create simulation from an image file"""
    dims = np.array([l1, l2, l3, gx, gy])
    input_abs = get_absolute_path(input)
    output_abs = get_absolute_path(output)
    sim.CreateDrawmatonSimulation(dims, input_abs, "image", output_abs, interactive=interactive)
    click.echo(f"Created simulation file from image: {output}")

@create.command()
@dimension_options
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input SVG file')
@click.option('--output', '-o', type=click.Path(), required=True, help='Output simulation file')
@click.option('--target-x', type=float, default=13.4, help='Target X position')
@click.option('--target-y', type=float, default=27.1, help='Target Y position')
@click.option('--target-w', type=float, default=16.3, help='Target width')
@click.option('--target-h', type=float, default=16.3, help='Target height')
def svg(l1, l2, l3, gx, gy, input, output, target_x, target_y, target_w, target_h):
    """Create simulation from an SVG file"""
    dims = np.array([l1, l2, l3, gx, gy])
    input_abs = get_absolute_path(input)
    output_abs = get_absolute_path(output)
    sim.CreateDrawmatonSimulation(dims, input_abs, "svg", output_abs)
    click.echo(f"Created simulation file from SVG: {output}")

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input simulation file')
def animate(input):
    """Show animation for a simulation file"""
    input_abs = get_absolute_path(input)
    sim.AnimateDrawmaton(input_abs)

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input simulation file')
@click.option('--output', '-o', type=click.Path(), required=True, help='Output GIF file')
def export_animation(input, output):
    """Export animation to a GIF file"""
    input_abs = get_absolute_path(input)
    output_abs = get_absolute_path(output)
    util.ExportAnimation(input_abs, output_abs)
    click.echo(f"Exported animation to: {output}")

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input simulation file')
@click.option('--output-bottom', '-b', type=click.Path(), required=True, help='Output bottom rotor SVG file')
@click.option('--output-top', '-t', type=click.Path(), required=True, help='Output top rotor SVG file')
def export_rotors(input, output_bottom, output_top):
    """Export rotor profiles as SVG files"""
    input_abs = get_absolute_path(input)
    output_bottom_abs = get_absolute_path(output_bottom)
    output_top_abs = get_absolute_path(output_top)
    util.ExportRotorsSVG(input_abs, output_bottom_abs, output_top_abs)
    click.echo(f"Exported rotor SVGs to: {output_bottom} and {output_top}")

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input simulation file')
def show_gaps(input):
    """Show rotor to base gaps"""
    input_abs = get_absolute_path(input)
    gap = util.CalculateRotorToBaseGap(input_abs)
    click.echo(f"Base distance: {gap[0]:.2f}")
    click.echo(f"Maximum rotor radius: {gap[1]:.2f}")
    click.echo(f"Minimum gap: {gap[2]:.2f}")

if __name__ == '__main__':
    cli()
