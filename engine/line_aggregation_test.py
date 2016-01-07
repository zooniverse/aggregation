from transcription import TranscriptionAPI

with TranscriptionAPI(245,"development") as project:
    project.__setup__()

    workflow_id = project.workflows.keys()[0]

    for subject_id,aggregations in project.__yield_aggregations__(workflow_id):
        print aggregations
        break