use crate::models::sdc1d::SDC;
use crate::system::SystemEnum;
use config::Config;

impl SystemEnum {
    pub fn from_config(config: &Config) -> Self {
        let system = config.get_string("model").unwrap();
        match system.as_str() {
            "sdc1d" => SystemEnum::SDC(SDC::from_config(config)),
            _ => panic!("Unknown system: {}", system),
        }
    }
}

impl SDC {
    pub fn from_config(config: &Config) -> Self {

        

        SDC {
            anchor_tiles: todo!(),
            strand_names: todo!(),
            glue_names: todo!(),
            scaffold: todo!(),
            strand_concentration: todo!(),
            glues: todo!(),
            glue_links: todo!(),
            colors: todo!(),
            kf: todo!(),
            g_se: todo!(),
            alpha: todo!(),
            friends_btm: todo!(),
            energy_bonds: todo!(),
        }
    }
}