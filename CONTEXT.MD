Hey we are basically building a out of the box ML/AI-Model Service for specific tasks.

Many companies and users have problems like:

- writing the best possible cold-email for them
- writing the best possible (with the highest reply rate) hinge outreach message
- ...

We want to provide people a service that they can call with instructions and context, and we will provide them an API after a couple minutes that is optimized for their specific task.

So our "black-box" looks something like this from the strucutre:

We have a GenerationModel (this function/agent is generating a message for you. This is also what you can later use in your product system via the API)
- parameters is C (the constant for your user/startup with all the context necessary) and p (the persona with all the context that you have about them. p is the person that you want to write a cold outreach mail or the profile that you want to reach out to on hinge)

We have a DataGeneration Function/Service
- this is generating our dataset with virtual P's (Personal) to train and validate our model on later
- It gets the C from the user, and some context about the domain where they are operating to generate the persona.

We have one DecisionMaker Agent. Our Virtual representation of the Decision Maker that we want to optimize for our real world system.
- Parameters are m (the message that is coming in), p the persona that it should have, and and object to determine what the response should be (for example for most use-cases it will be a true/false binary classification if it will reply or not but might be something else too with more possible labels)


Later on we will build a system that is trying to self improve based on the outcome of your first run on the generated dataset. Evaulating the accuracy that it had on the DecisionMaker.