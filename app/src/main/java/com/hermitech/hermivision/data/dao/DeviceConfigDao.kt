package com.hermitech.hermivision.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.hermitech.hermivision.data.model.DeviceConfigEntity

@Dao
interface DeviceConfigDao {

    @Query("SELECT * FROM device_config WHERE id = 1 LIMIT 1")
    suspend fun getConfig(): DeviceConfigEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertConfig(config: DeviceConfigEntity)
}