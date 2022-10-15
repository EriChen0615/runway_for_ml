import click
import shutil
import os
import pathlib

@click.group()
def cli():
    pass

@click.command()
@click.option('--dest', '-d', 'dest_dir',
    required=True, 
    type=str,
    default=os.getcwd())
def init(dest_dir):
    assert not os.path.exists('configs/'), """
        configs/ folder exist. You may be overwriting an already initialized
        runway project. Rename or delete configs/ folder and try again if you are sure.
        """
    click.echo("Initializing...")
    runway_dir = pathlib.Path(__file__).parent.resolve() # get parent directory
    click.echo(runway_dir)
    shutil.copytree(os.path.join(runway_dir, 'configs'), os.path.join(dest_dir, 'configs/'))
    click.echo("Project initialized. Welcome to Runway!")

@click.command('hi')
def hi():
    click.echo("hello from runway!")

cli.add_command(init)
cli.add_command(hi)

if __name__ == '__main__':
    cli()
    
