from numba import jit

@jit(nopython=True)
def jit_calculate_reward(
    fti,
    xti,
    gti,
    hti,
    ft_gti,
    xt_gti,
    gt_gti,
    ht_gti,
    irradiance,
    panel_area,
    panel_efficiency,
    backlog,
    backlog_gti,
    e_idle,
    e_frame,
    e_tx_rx,
    battery_level,
    battery_capacity,
    processing_rate,
    proc_interval,
    agent_id,
    w
    ):

    # if(fti + hti > processing_rate):
    #     k = processing_rate / (fti + hti)
    #     fti *= k
    #     hti *= k
        
    #     # hti = processing_rate - fti
        
    #     # if(hti > fti):
    #     #     fti = processing_rate - hti
    #     # else:
    #     #     hti = processing_rate - fti
        
    # if(ft_gti + ht_gti > processing_rate):
    #     k = processing_rate / (ft_gti + ht_gti)
    #     ft_gti *= k
    #     ht_gti *= k
        # ht_gti = processing_rate - ft_gti

        # if(ht_gti > ft_gti):
        #     ft_gti = processing_rate - ht_gti
        # else:
        #     ht_gti = processing_rate - ft_gti
    
    if(fti + hti) > processing_rate:
        hti = processing_rate - fti
                
    if(ft_gti + ht_gti) > processing_rate:
        ht_gti = processing_rate - ft_gti
    
    # max_backlog = proc_interval * processing_rate * 10
    
    panel_energy = irradiance * panel_area * panel_efficiency * proc_interval
    actual_battery = battery_level + panel_energy
    
    processable = max(min(backlog, int((actual_battery - e_idle) / e_frame), processing_rate * proc_interval), 0)    
    # processed = min(processable, fti * proc_interval)
    
    # needed_energy = processed * e_frame + e_idle
    # needed_energy = (fti * proc_interval * e_frame) + e_idle
    
    needed_energy = (fti * proc_interval * e_frame) + e_idle
        
    local_reward = 0
    offloading_reward = 0
    
    if(actual_battery > needed_energy and processable > 0):
        processed = min(fti * proc_interval, processable)
        local_reward = (processed/processable) + (actual_battery/battery_capacity) + (processed / backlog)
        
        actual_battery = max(actual_battery - needed_energy, 0)
        backlog = max(backlog - processed, 0)
        # if(backlog > 0):
        #     local_reward = (processed/processable) * ((actual_battery - needed_energy)/battery_capacity) * 100 
        # else:
        #     local_reward = (processed / processable) * (actual_battery / battery_capacity) * processed
        
    else:
        local_reward = 0
        if(processable == 0 and fti == 0):
            local_reward = actual_battery / battery_capacity
            
        actual_battery = max(actual_battery - e_idle, 0)

    # else:
    #     if(fti > 0):
    #         local_reward = 0
    #     else:
    #         # local_reward =  (actual_battery/battery_capacity) 
    #         if(backlog > 0):
    #             local_reward =  (actual_battery/battery_capacity) * (max_backlog / backlog) 
    #         else:
    #             local_reward =  (actual_battery/battery_capacity) * max_backlog
                
    #     actual_battery = max(actual_battery - e_idle, 0)
            
        
    remaining_framerate = processing_rate - fti
    
    if(remaining_framerate > 0 and xti != 0):
        
        if ft_gti + ht_gti > processing_rate:
            return local_reward
        
        if(xti == 1 and gti != agent_id and hti > 0 and xt_gti == 2 and gt_gti == agent_id and ht_gti > 0 and backlog_gti == 0):
            ht = min(hti, ht_gti)
            
            # needed_energy = ht * self._proc_interval * self.e_tx_rx
            processable = max(min(backlog, int((actual_battery - e_idle) / e_tx_rx), remaining_framerate * proc_interval), 0)
            # if(processable > 0):
            processed = min(ht * proc_interval, processable)
            needed_energy = ht * proc_interval * e_tx_rx
            
            if(needed_energy <= actual_battery and processable > 0):
                # actual_battery = max(actual_battery - needed_energy, 0)
                # backlog = max(backlog - processed, 0)
                offloading_reward = (processed/(processable)) + (actual_battery/battery_capacity) + (processed / backlog)

                # if(backlog > 0):
                #     offloading_reward = (processed/(processable)) * (actual_battery/battery_capacity) * (processed / backlog) 
                # else:
                    # offloading_reward = (processed/(processable)) * (actual_battery/battery_capacity) * processed 
            else:
                offloading_reward = 0
                if(processable == 0 and ht == 0):
                    offloading_reward = actual_battery / battery_capacity
            # else:
            #     if(ht > 0):
            #         offloading_reward = 0
            #     else:
            #         # offloading_reward =  (actual_battery/battery_capacity) 
            #         if(backlog > 0):
            #             offloading_reward =  (actual_battery/battery_capacity) * (max_backlog / backlog) 
            #         else:
            #             offloading_reward =  (actual_battery/battery_capacity) * max_backlog
            
        elif(xti == 2 and gti != agent_id and hti > 0 and xt_gti == 1 and gt_gti == agent_id and ht_gti > 0 and backlog == 0):
            ht = min(hti, ht_gti)
            # needed_energy = ht * self._proc_interval * self.e_tx_rx
            processable = max(min(backlog, int((actual_battery - e_idle) / (e_tx_rx + e_frame)), remaining_framerate * proc_interval), 0)
            # if(processable > 0):
            processed = min(ht * proc_interval, processable)
            needed_energy = ht * proc_interval * (e_tx_rx + e_frame)
            
            if(needed_energy <= actual_battery and processable > 0):
                # actual_battery = max(actual_battery - needed_energy, 0)
                # backlog = max(backlog - processed, 0)
                offloading_reward = (processed/(processable)) + (actual_battery/battery_capacity)

                # if(backlog > 0):
                #     offloading_reward = (processed/(processable)) * (actual_battery/battery_capacity) * (processed / backlog) 
                # else:
                    # offloading_reward = (processed/(processable)) * (actual_battery/battery_capacity) * processed
            else:
                offloading_reward = 0
                if(processable == 0 and ht == 0):
                    offloading_reward = actual_battery / battery_capacity
            # else:
            #     if(ht > 0):
            #         offloading_reward = 0
            #     else:
            #         # offloading_reward =  (actual_battery/battery_capacity) 
            #         if(backlog > 0):
            #             offloading_reward =  (actual_battery/battery_capacity) * (max_backlog / backlog) 
            #         else:
            #             offloading_reward =  (actual_battery/battery_capacity) * max_backlog
                        
    return local_reward + w * offloading_reward
                        
    # if(actual_battery <= needed_energy):
    #     return 000
        
    # actual_battery = max(actual_battery - needed_energy, 0)

    # if(processable > 0):
    #     # if(backlog > 0):
    #     # local_reward = (processed / processable) * (actual_battery / battery_capacity)
    #     local_reward = w * (processed / processable) * (actual_battery / battery_capacity) * (processed / backlog)
    #     backlog = max(backlog - processed, 0)
    # else:
    #     return 0
    
    # if(xti == 0):
    #     return local_reward
    
    # remaining_framerate = processing_rate - fti
    # if(remaining_framerate < 0):
    #     return local_reward
    
    # offloading_reward = 0
    
    # if(xti == 1 and gti != agent_id and hti > 0 and xt_gti == 2 and gt_gti == agent_id and ht_gti > 0):
    #     ht = min(hti, ht_gti)
    #     # processable = min(backlog, int(actual_battery / e_frame))
    #     processable = max(min(backlog, int(actual_battery / e_txrx), remaining_framerate * proc_interval), 0)
    #     processed = min(processable, ht * proc_interval)
        
    #     needed_energy = processed * e_txrx
    #     # needed_energy = ht * proc_interval * e_txrx
        
    #     if(actual_battery > needed_energy):
    #         if(processable > 0):
    #             # offloading_reward = float(processed/processable) * (actual_battery / battery_capacities[agent_id])
    #             offloading_reward = w * float(processed/processable) * (actual_battery / battery_capacity) * (processed / backlog)
    #             # offloading_reward = float(processed/processable) * (actual_battery / battery_capacity)
    #         # else:
    #         #     offloading_reward = 0
    #     else:
    #         # return 000
    #         # offloading_reward = 0
    #         offloading_reward = 000
    
    # elif(xti == 2 and gti != agent_id and hti > 0 and xt_gti == 1 and gt_gti == agent_id and ht_gti > 0):
    #     ht = min(hti, ht_gti)
    #     # processable = min(backlog, int(actual_battery / e_frame))
    #     processable = max(min(backlog, int(actual_battery / (e_txrx + e_frame)), remaining_framerate * proc_interval), 0)
    #     processed = min(processable, ht * proc_interval)
        
    #     needed_energy = processed * (e_frame + e_txrx)
    #     # needed_energy = ht * proc_interval * (e_txrx + e_frame)
        
    #     if(actual_battery > needed_energy):
    #         if(processable > 0):
    #             # offloading_reward = float(processed/processable) * (actual_battery / battery_capacities[agent_id])
    #             offloading_reward = w * float(processed/processable) * (actual_battery / battery_capacity) * (processed / backlog)
    #             # offloading_reward = float(processed/processable) * (actual_battery / battery_capacity)
    #         # else:
    #         #     offloading_reward = 0
    #     else:
    #         # return 000
    #         # offloading_reward = 0
    #         offloading_reward = 000
    
    # return local_reward + (offloading_reward)
        