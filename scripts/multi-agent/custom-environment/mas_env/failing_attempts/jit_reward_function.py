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
    e_idle,
    e_frame,
    e_txrx,
    battery_level,
    battery_capacity,
    processing_rate,
    proc_interval,
    agent_id,
    w
    ):
    
    if(fti + hti > processing_rate):
        k = processing_rate / (fti + hti)
        fti *= k
        hti *= k
        
        # hti = processing_rate - fti
        
        # if(hti > fti):
        #     fti = processing_rate - hti
        # else:
        #     hti = processing_rate - fti
        
    if(ft_gti + ht_gti > processing_rate):
        k = processing_rate / (ft_gti + ht_gti)
        ft_gti *= k
        ht_gti *= k
        # ht_gti = processing_rate - ft_gti

        # if(ht_gti > ft_gti):
        #     ft_gti = processing_rate - ht_gti
        # else:
        #     ht_gti = processing_rate - ft_gti
        
    panel_energy = irradiance * panel_area * panel_efficiency * proc_interval
    actual_battery = battery_level + panel_energy
    
    processable = max(min(backlog, int((actual_battery - e_idle) / e_frame), processing_rate * proc_interval), 0)    
    processed = min(processable, fti * proc_interval)
    
    needed_energy = processed * e_frame + e_idle
    # needed_energy = (fti * proc_interval * e_frame) + e_idle
    
    local_reward = 0

    if(actual_battery <= needed_energy):
        return -1
        
    actual_battery = max(actual_battery - needed_energy, 0)

    if(processable > 0):
        # if(backlog > 0):
        # local_reward = (processed / processable) * (actual_battery / battery_capacity)
        local_reward = w * (processed / processable) * (actual_battery / battery_capacity) * (processed / backlog)
        backlog = max(backlog - processed, 0)
    else:
        return 0
    
    if(xti == 0):
        return local_reward
    
    remaining_framerate = processing_rate - fti
    if(remaining_framerate < 0):
        return local_reward
    
    offloading_reward = 0
    
    if(xti == 1 and gti != agent_id and hti > 0 and xt_gti == 2 and gt_gti == agent_id and ht_gti > 0):
        ht = min(hti, ht_gti)
        # processable = min(backlog, int(actual_battery / e_frame))
        processable = max(min(backlog, int(actual_battery / e_txrx), remaining_framerate * proc_interval), 0)
        processed = min(processable, ht * proc_interval)
        
        needed_energy = processed * e_txrx
        # needed_energy = ht * proc_interval * e_txrx
        
        if(actual_battery > needed_energy):
            if(processable > 0):
                # offloading_reward = float(processed/processable) * (actual_battery / battery_capacities[agent_id])
                offloading_reward = w * float(processed/processable) * (actual_battery / battery_capacity) * (processed / backlog)
                # offloading_reward = float(processed/processable) * (actual_battery / battery_capacity)
            # else:
            #     offloading_reward = 0
        else:
            # return -1
            # offloading_reward = 0
            offloading_reward = -1
    
    elif(xti == 2 and gti != agent_id and hti > 0 and xt_gti == 1 and gt_gti == agent_id and ht_gti > 0):
        ht = min(hti, ht_gti)
        # processable = min(backlog, int(actual_battery / e_frame))
        processable = max(min(backlog, int(actual_battery / (e_txrx + e_frame)), remaining_framerate * proc_interval), 0)
        processed = min(processable, ht * proc_interval)
        
        needed_energy = processed * (e_frame + e_txrx)
        # needed_energy = ht * proc_interval * (e_txrx + e_frame)
        
        if(actual_battery > needed_energy):
            if(processable > 0):
                # offloading_reward = float(processed/processable) * (actual_battery / battery_capacities[agent_id])
                offloading_reward = w * float(processed/processable) * (actual_battery / battery_capacity) * (processed / backlog)
                # offloading_reward = float(processed/processable) * (actual_battery / battery_capacity)
            # else:
            #     offloading_reward = 0
        else:
            # return -1
            # offloading_reward = 0
            offloading_reward = -1
    
    return local_reward + (offloading_reward)
        