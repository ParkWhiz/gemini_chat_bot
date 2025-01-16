# Instructions

## Pre-Requisites
Docker and Chrome/Firefox.  Edge? Maybe, idk.

## Get Gemini API Key
You'll need this as an arg to start the service

## Run via Docker
For convenience the script `run_service.sh` can be leveraged.
```./run_service.sh GEMINI_API_KEY YOUR_CODE_DIRECTORY```

## Alternate Setup
### Directories in scope
There is a set of directories normally meant to contain test data or other fixtures that are often large and can cause projects to grow past the context window size (think `vendor`, `node_modules`, `spec`).  There is a normal set of files that are excluded in the code.  There are defined in `read_project_files`
```def read_project_files(
        exclude_dirs=['.github', '.git', '.cm', '.idea', 'webpack', 'spec', 'script', 'benchmarks', 'bin', 'benchmarks', 'log', 'node_modules', 'dist', 'fixtures', 'vendor']
):```

These can either be overridden OR added to using a `exclude_dirs.json` file in the root of the volume that is mounted in the container.  There is a sample file `.exclude_dirs_example.json` in the root of the project.  It is a simple JSON document.

```
{
	"override": false,
	"exclude_dirs": ["db", "hot_fix"]
}
```

### Files in scope
Files are read on a whitelist basis by extension.  Like directories there is a default list that is considered.  This list is defined in `read_project_files` 

```            if file.endswith(('.py', '.json', '.kt', '.html', '.js', '.cs', '.qml', '.asp', '.vb',
                              '.ts', '.java', '.c', '.cpp', '.h', '.hpp', '.go', '.rs', '.swift',
                              '.sh', '.rb', '.php', '.mdx', '.rs', '.sql')):
```

This list can be extended or overridden by a `.include_extensions.json` file in the root of the volume that is mounted in the container.   There is a sample file `.include_extensions_example.json` in the root of the project.  It is a simple JSON document.

```
{
	"override": false,
	"include_extensions": [".cpp", ".myFile"]
}
```

## Open Chat Bot and submit a request.
Navigate to `http://127.0.0.1:5000/index.html`

Enter your query, hit enter/submit, be patient, especially on a large code base.   The first upload will generate a context cache with your code base context so subsequent queries will be speedier.

Note: If you are on MacOS Monterey or later and have AirPlay enabled, then you will be unable to use port 5000. You must either disable AirPlay or run this project on a different port.
