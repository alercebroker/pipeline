import typer
import cli_build
import cli_deploy
import cli_update


app = typer.Typer()
app.add_typer(cli_build.app, name="build")
app.add_typer(cli_deploy.app, name="deploy")
app.add_typer(cli_update.app, name="update")

if __name__ == "__main__":
    app()