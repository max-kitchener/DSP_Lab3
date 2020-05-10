import os
from git import Repo

# Directory for remote 'origin' for new project
remote_dir = 'H:\\Documents\\Repos'

project_exists = False

# Queries used for name of project, and creates a directory for it in the working folder
while False == project_exists:
    # Fetch current working directory
    local_dir =os.getcwd()
    cur_dir = local_dir.split("\\")
    # Use the current folder as name for repo folder
    repo_name = cur_dir[len(cur_dir)-1]

    # Setup directory for remote repo
    remote_dir = remote_dir + '\\' + repo_name
    os.mkdir(remote_dir)

    # initialise remote as a bare repository
    remote_repo = Repo.init(remote_dir, bare=True)
    assert remote_repo.bare  # is repo bare?
    # initializes local repository
    local_repo = Repo.init(local_dir, bare=False)
    assert not local_repo.bare  # repo should not be bare

     # Create README file and briefly populate
    project_readme = local_dir + '\\README.md'
    with open(project_readme, 'a+') as file:
        # Write title
        file.write('# ' + repo_name)
        # query user for brief description of project
        file.write('\n' + input('Insert a brief description here\n'))
    # with

    # Add README to repository and commit
    local_repo.index.add([project_readme])
    local_repo.index.commit('Initial Commit')

    # Set the remote_repo as the remote 'origin' for local_repo
    origin = local_repo.create_remote('origin', remote_dir)

    # Push 'Initial Commit' to origin
    # REVIEW Need to push master branch of local repo to remote
    origin.push('master')

    # Verify origin
    assert origin.exists(), 'remote origin not found'
    assert origin == local_repo.remotes.origin == local_repo.remotes['origin'], 'remote origin not found'

# If

#End of file
