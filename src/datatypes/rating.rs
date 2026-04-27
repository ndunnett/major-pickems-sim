#[nutype::nutype(
    validate(greater = 0),
    derive(Debug, Display, Clone, Copy, Serialize, Deserialize,)
)]
pub struct Rating(u16);

impl Rating {
    #[must_use]
    pub fn to_f32(self) -> f32 {
        f32::from(unsafe { std::mem::transmute::<Self, u16>(self) })
    }
}
