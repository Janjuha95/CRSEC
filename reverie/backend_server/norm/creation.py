import json
import sys
sys.path.append('../')
from utils import *
from norm.normDatabase import *
from llm_router import llm_call


class Creation:
    def __init__(self, system_msg, model="gpt-3.5-turbo-16k", temprature=1, max_tokens=4096, top_p=1,
                 frequency_penalty=0, presence_penalty=0):
        # parameter
        self.msg = [{"role": "system", "content": system_msg}]
        self.model = model
        self.temp = temprature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    # input prompt, return explicit norms
    def creation(self, prompt, agent_description):
        '''
        To create 5 norms for the agent.
        The number of norms can be adjusted in the usr_prompt_v6.txt
            which is located in:
            ./creation_prompt
        Args:
            prompt: user prompt
            agent_description: agent's scratch
        Returns:
            norms in json format
        '''
        user_prompt = {"role": "user", "content": prompt}
        self.msg.append(user_prompt)

        agent_prompt = {"role": "user", "content": agent_description}
        self.msg.append(agent_prompt)

        composed = "\n\n".join(m["content"] for m in self.msg)
        return llm_call(composed, call_type="norm_creation")


def Create(rs):
    while 1:
        choice = input("Regenerate norms? (y or n): ").strip()

        if choice == 'y':
            with open('./norm/creation_prompt/sys_prompt.txt', 'rt', encoding='utf-8') as f:
                sys_prompt = f.read()
                #print("sys_prompt:", sys_prompt)
            with open('./norm/creation_prompt/usr_prompt_v6.txt', 'rt', encoding='utf-8') as f:
                usr_prompt = f.read()
                #print('usr_prompt:', usr_prompt)

            entrepreneur = input("Enter the name of the entrepreneur: ").strip()

            if entrepreneur in rs.personas:
                create_bot = Creation(sys_prompt)

                agent_scratch = f"{fs_storage}/{rs.sim_code}/personas/{entrepreneur}/bootstrap_memory/scratch.json"
                with open(agent_scratch, 'rt', encoding='utf-8') as f:
                    agent_description = f.read()
                    #print('agent_description:', agent_description)

                response = create_bot.creation(usr_prompt, agent_description)
                print(response)
                
                norm_seed_file = f"{fs_storage}/{rs.sim_code}/personas/{entrepreneur}/norms/personal_norm_database_validity.json"
                with open(norm_seed_file, 'w', encoding='utf-8') as fw:
                    json.dump(json.loads(response), fw, ensure_ascii=False)
                    pass
                rs.personas[entrepreneur].scratch.norm_count = 5
                norm_file = f"{fs_storage}/{rs.sim_code}/personas/{entrepreneur}/norms/personal_norm_database.json"
                with open(norm_file, 'w', encoding='utf-8') as fw:
                    json.dump(json.loads(response), fw, ensure_ascii=False)
                    pass
                rs.personas[entrepreneur].scratch.act_norm_count = 5
                norm_saved = f"{fs_storage}/{rs.sim_code}/personas/{entrepreneur}/norms"
                rs.personas[entrepreneur].norm_database = NormDatabase(norm_saved,rs.personas[entrepreneur].scratch.norm_count,rs.personas[entrepreneur].scratch.act_norm_count,rs.personas[entrepreneur])
            else:
                print(entrepreneur, ": This person is not exist!")
        elif choice == 'n':
            return
