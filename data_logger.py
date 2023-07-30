def log_lunar_lander(lunar_lander, output_file_path, num_episodes):
   agent = lunar_lander
   output_file = output_file_path
   with open(output_file,"w") as the_file:
      # Discount Rate:
      the_file.write("Discount Factor          : "+str(agent.gamma)+"\n")
      # Learning Rate:
      the_file.write("Learning Rate            : "+str(agent.alpha)+"\n")
      # Epsilon Schedule:
      the_file.write("Epsilon Schedule Type    : "+agent.epsilson_schedule_type+"\n")
      # Number of Hidden Neural Network Cells per hidden layer:
      the_file.write("Hidden Layer Cell Count  : "+str(agent.hidden_sizes[0])+"\n")
      # Number of Episodes:
      the_file.write("Number of Episodes       : "+str(num_episodes)+"\n")
      # Batch Size
      the_file.write("Batch Size               : "+str(agent.batch_size)+"\n")
      # Write the meta data:
      the_file.write("Accumulated Reward       : %s\n"%' '.join(map(str, agent.total_reward_vector)))
      the_file.write("Epsilon Schedule         : %s\n"%' '.join(map(str, agent.epsilon_schedule)))