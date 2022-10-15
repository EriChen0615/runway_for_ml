import click
import shutil
import os

@click.command()
@click.option('--dest', '-d', 'dest_dir',
    require=True, 
    type=str,
    default=os.getcwd())
def init(dest_dir):
    assert not os.path.exists('configs/') """
        configs/ folder exist. You may be overwriting an already initialized
        runway project. Rename or delete configs/ folder and try again if you are sure.
        """
    shutil.copytree('./configs/', os.path.join(dest_dir, 'configs/'))
    click.echo("Project initialized. Welcome to Runway!")
    
